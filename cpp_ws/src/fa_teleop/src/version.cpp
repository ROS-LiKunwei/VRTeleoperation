#include "bundle_version.hpp"
#include "fa_teleop/core.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>

namespace fa {
std::string file_sha256(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Missing bundle file: " + path);
  EVP_MD_CTX *raw = EVP_MD_CTX_new();
  if (!raw)
    throw std::runtime_error("SHA256 allocation failed");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(raw,
                                                              EVP_MD_CTX_free);
  if (EVP_DigestInit_ex(raw, EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA256 init failed");
  std::array<char, 65536> buffer{};
  while (file) {
    file.read(buffer.data(), buffer.size());
    if (file.gcount() > 0 &&
        EVP_DigestUpdate(raw, buffer.data(), file.gcount()) != 1)
      throw std::runtime_error("SHA256 update failed");
  }
  if (!file.eof())
    throw std::runtime_error("Bundle read failed: " + path);
  unsigned char bytes[EVP_MAX_MD_SIZE];
  unsigned int n = 0;
  if (EVP_DigestFinal_ex(raw, bytes, &n) != 1)
    throw std::runtime_error("SHA256 final failed");
  std::ostringstream s;
  s << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < n; ++i)
    s << std::setw(2) << static_cast<unsigned int>(bytes[i]);
  return s.str();
}
const char *release_id() { return kReleaseId; }
std::string deployment_id(const Config &config) {
  return std::string(kReleaseId) + ":" + config.profile_sha;
}
std::string verify_bundle(const std::string &directory) {
  namespace fs = std::filesystem;
  const fs::path root = fs::canonical(directory);
  if (file_sha256((root / "release.json").string()) != kReleaseId)
    throw std::runtime_error(
        "Compiled release does not match deployment manifest");
  if (file_sha256((root / "files.sha256").string()) != kFileListSha)
    throw std::runtime_error(
        "Compiled file list does not match deployment manifest");
  std::ifstream input(root / "files.sha256");
  std::string line;
  size_t count = 0;
  while (std::getline(input, line)) {
    if (line.size() < 67 || line.substr(64, 2) != "  ")
      throw std::runtime_error("Invalid bundle file list");
    fs::path relative = line.substr(66);
    if (relative.is_absolute())
      throw std::runtime_error("Absolute bundle entry");
    for (const auto &part : relative)
      if (part == "..")
        throw std::runtime_error("Escaping bundle entry");
    const auto path = root / relative;
    if (file_sha256(path.string()) != line.substr(0, 64))
      throw std::runtime_error("Version mismatch: " + relative.string());
    ++count;
  }
  if (count < 10)
    throw std::runtime_error("Incomplete deployment bundle");
  return kReleaseId;
}
} // namespace fa

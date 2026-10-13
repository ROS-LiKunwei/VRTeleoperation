import xml.etree.ElementTree as ET
import os


def check_symmetric_links_mass(urdf_file, mass_tolerance=0.01, verbose=False):
    """
    检查URDF文件中左右对称link的质量是否接近

    参数:
        urdf_file: URDF文件路径
        mass_tolerance: 质量允许的差异容忍度 (默认0.01)
        verbose: 是否打印详细信息 (默认False)
    """

    # 检查文件是否存在
    if not os.path.exists(urdf_file):
        print(f"错误: 文件 '{urdf_file}' 不存在")
        return []

    # 解析URDF文件
    try:
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        print(f"成功解析URDF文件: {urdf_file}")
    except Exception as e:
        print(f"错误: 无法解析URDF文件: {e}")
        return []

    # 创建link名称到质量的映射
    link_mass_map = {}
    links_without_mass = []

    # 遍历所有link元素
    for link in root.findall("link"):
        link_name = link.get("name")

        # 查找inertial元素
        inertial = link.find("inertial")
        if inertial is not None:
            # 查找mass元素
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                try:
                    mass = float(mass_elem.get("value"))
                    link_mass_map[link_name] = mass
                    if verbose:
                        print(f"Link '{link_name}': mass = {mass}")
                except (ValueError, TypeError):
                    if verbose:
                        print(f"警告: 无法解析link '{link_name}'的质量值")
            else:
                links_without_mass.append(link_name)
        else:
            links_without_mass.append(link_name)

    if verbose and links_without_mass:
        print(f"\n以下link没有质量信息: {links_without_mass}")

    # 查找左右对称的link对
    left_links = [name for name in link_mass_map.keys() if name.startswith("left_")]
    right_links = [name for name in link_mass_map.keys() if name.startswith("right_")]

    print(f"找到 {len(left_links)} 个左link, {len(right_links)} 个右link")

    # 检查对称link对
    mismatched_pairs = []
    matched_pairs = []

    for left_link in left_links:
        # 从left_* 生成对应的right_*
        right_link = left_link.replace("left_", "right_", 1)

        if right_link in link_mass_map:
            left_mass = link_mass_map[left_link]
            right_mass = link_mass_map[right_link]
            mass_diff = abs(left_mass - right_mass)

            if mass_diff > mass_tolerance:
                mismatched_pairs.append(
                    {
                        "left_link": left_link,
                        "right_link": right_link,
                        "left_mass": left_mass,
                        "right_mass": right_mass,
                        "mass_diff": mass_diff,
                    }
                )
            else:
                matched_pairs.append(
                    {
                        "left_link": left_link,
                        "right_link": right_link,
                        "left_mass": left_mass,
                        "right_mass": right_mass,
                        "mass_diff": mass_diff,
                    }
                )

    # 输出结果
    if mismatched_pairs:
        print(f"\n发现 {len(mismatched_pairs)} 对质量不匹配的对称link:")
        print("-" * 90)
        print(
            f"{'左Link':<25} {'右Link':<25} {'左质量':<10} {'右质量':<10} {'差异':<10}"
        )
        print("-" * 90)

        for pair in mismatched_pairs:
            print(
                f"{pair['left_link']:<25} {pair['right_link']:<25} "
                f"{pair['left_mass']:<10.4f} {pair['right_mass']:<10.4f} "
                f"{pair['mass_diff']:<10.4f}"
            )
    else:
        print(f"\n✓ 所有对称link的质量差异都在容忍度({mass_tolerance})范围内")

        if verbose and matched_pairs:
            print(f"匹配的对称link对 ({len(matched_pairs)} 对):")
            for pair in matched_pairs:
                print(
                    f"  {pair['left_link']} <-> {pair['right_link']}: "
                    f"{pair['left_mass']:.4f} vs {pair['right_mass']:.4f} "
                    f"(差异: {pair['mass_diff']:.4f})"
                )

    return mismatched_pairs


# 使用示例
if __name__ == "__main__":
    # 在这里直接指定文件和参数
    urdf_file = "../urdf/fa_robot.urdf"

    tolerance = 0.001
    verbose_mode = True

    print(f"开始检查URDF文件: {urdf_file}")
    print(f"质量容忍度: {tolerance}")
    print(f"详细模式: {verbose_mode}")
    print("=" * 60)

    # 执行检查
    mismatched = check_symmetric_links_mass(
        urdf_file=urdf_file, mass_tolerance=tolerance, verbose=verbose_mode
    )

    print("\n" + "=" * 60)
    print("检查完成!")

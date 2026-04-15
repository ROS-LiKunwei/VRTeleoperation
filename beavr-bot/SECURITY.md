# Security Policy

## Supported Versions

We actively support the most recent **minor release** of BeaVR. Older versions are **not guaranteed** to receive security updates.

| Version   | Supported          |
| --------- | ------------------ |
| main (dev branch) | :white_check_mark: |
| latest tagged release | :white_check_mark: |
| older releases (< 1.0) | :x:                |

If you are using BeaVR in research or production, we strongly recommend keeping your installation updated to the latest release.

---

## Reporting a Vulnerability

We take the security of BeaVR seriously. If you discover a potential security issue (for example, related to networking, process management, or unsafe data handling), please **do not open a public GitHub issue**.

Instead, please report it privately by emailing:

**beavr-security@mit.edu** (replace with your team email/contact)

### What to include
- A description of the issue.
- Steps to reproduce, if possible.
- Any potential impact or severity assessment.
- Suggested fixes or patches (if you have one).

### What to expect
- We will acknowledge receipt of your report within **5 business days**.
- You will receive updates as we investigate and address the issue.
- Once the issue is resolved, we will publish an advisory and update the repository accordingly.
- If appropriate, we will credit you for the discovery (unless you prefer to remain anonymous).

---

## Scope

BeaVR is primarily a **research and educational project**. While we strive for secure practices, we cannot guarantee enterprise-level support. Vulnerability reports are welcome, especially regarding:
- Teleoperation network layer (ZMQ communication, serialization).
- Process management and inter-process communication.
- Configuration handling (e.g. YAML/CLI overrides).
- Data storage and logging.

---

## Responsible Disclosure

We kindly ask that you follow responsible disclosure practices:
- Report privately first.
- Do not share details publicly until a fix has been released and coordinated with the maintainers.

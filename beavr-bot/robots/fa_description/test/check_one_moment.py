import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple, Optional


def compare_left_right_inertial_xyz(urdf_path: str, tolerance: float = 1e-6) -> Dict:
    """
    比较URDF文件中左右对称link的惯性一阶矩(xyz)

    参数:
        urdf_path: URDF文件路径
        tolerance: 数值比较容差

    返回:
        Dict: 包含比较结果的字典
    """

    def parse_xyz(xyz_str: str) -> List[float]:
        """解析xyz字符串为浮点数列表"""
        try:
            return [float(x) for x in xyz_str.strip().split()]
        except (ValueError, AttributeError):
            return None

    def are_equal(a: float, b: float) -> bool:
        """比较两个浮点数是否相等（考虑容差）"""
        return abs(a - b) <= tolerance

    def are_opposite(a: float, b: float) -> bool:
        """比较两个浮点数是否为相反数（考虑容差）"""
        return abs(a + b) <= tolerance

    # 解析URDF文件
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        return {"error": f"解析URDF文件失败: {e}"}

    # 收集所有link及其惯性参数
    links_data = {}

    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue

        inertial = link.find("inertial")
        if inertial is None:
            continue

        origin = inertial.find("origin")
        if origin is None:
            continue

        xyz_str = origin.get("xyz")
        if not xyz_str:
            continue

        xyz = parse_xyz(xyz_str)
        if xyz is None or len(xyz) != 3:
            continue

        links_data[link_name] = {"xyz": xyz, "x": xyz[0], "y": xyz[1], "z": xyz[2]}

    # 分离左右link并匹配
    left_links = {}
    right_links = {}

    for link_name, data in links_data.items():
        if link_name.startswith("left_"):
            suffix = link_name[5:]  # 移除"left_"前缀
            left_links[suffix] = data
        elif link_name.startswith("right_"):
            suffix = link_name[6:]  # 移除"right_"前缀
            right_links[suffix] = data

    # 找出共同的后缀（左右配对的link）
    common_suffixes = set(left_links.keys()) & set(right_links.keys())

    results = {
        "comparisons": [],
        "summary": {"total_pairs": len(common_suffixes), "passed": 0, "failed": 0},
    }

    # 比较每对左右link
    for suffix in sorted(common_suffixes):
        left_data = left_links[suffix]
        right_data = right_links[suffix]

        comparison = {
            "suffix": suffix,
            "left_link": f"left_{suffix}",
            "right_link": f"right_{suffix}",
            "left_xyz": left_data["xyz"],
            "right_xyz": right_data["xyz"],
            "checks": {},
        }

        # 检查x是否一致
        x_check = are_equal(left_data["x"], right_data["x"])
        comparison["checks"]["x_equal"] = {
            "result": x_check,
            "left_x": left_data["x"],
            "right_x": right_data["x"],
        }

        # 检查y是否为相反数
        y_check = are_opposite(left_data["y"], right_data["y"])
        comparison["checks"]["y_opposite"] = {
            "result": y_check,
            "left_y": left_data["y"],
            "right_y": right_data["y"],
        }

        # 检查z是否一致
        z_check = are_equal(left_data["z"], right_data["z"])
        comparison["checks"]["z_equal"] = {
            "result": z_check,
            "left_z": left_data["z"],
            "right_z": right_data["z"],
        }

        # 总体是否通过
        all_passed = x_check and y_check and z_check
        comparison["all_passed"] = all_passed

        if all_passed:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1

        results["comparisons"].append(comparison)

    # 添加统计信息
    results["statistics"] = {
        "total_links_with_inertial": len(links_data),
        "left_links_found": len(left_links),
        "right_links_found": len(right_links),
        "matched_pairs": len(common_suffixes),
    }

    return results


def print_comparison_results(results: Dict, verbose: bool = False):
    """
    打印比较结果

    参数:
        results: compare_left_right_inertial_xyz函数的返回结果
        verbose: 是否显示详细信息
    """

    if "error" in results:
        print(f"错误: {results['error']}")
        return

    summary = results["summary"]
    stats = results["statistics"]

    print("=" * 60)
    print("URDF左右对称Link惯性一阶矩比较结果")
    print("=" * 60)
    print(f"\n统计信息:")
    print(f"  - 总link数(含惯性参数): {stats['total_links_with_inertial']}")
    print(f"  - 左侧link数: {stats['left_links_found']}")
    print(f"  - 右侧link数: {stats['right_links_found']}")
    print(f"  - 匹配的左右对: {stats['matched_pairs']}")
    print(f"\n比较结果:")
    print(f"  - 总对数: {summary['total_pairs']}")
    print(f"  - 通过: {summary['passed']}")
    print(f"  - 失败: {summary['failed']}")

    if summary["failed"] > 0 or verbose:
        print(f"\n详细比较结果:")

        for comp in results["comparisons"]:
            if not comp["all_passed"] or verbose:
                print(f"\n[{comp['suffix']}]")
                print(f"  左: {comp['left_link']} = {comp['left_xyz']}")
                print(f"  右: {comp['right_link']} = {comp['right_xyz']}")

                checks = comp["checks"]
                if not checks["x_equal"]["result"]:
                    print(
                        f"  ✗ X不一致: 左={checks['x_equal']['left_x']}, 右={checks['x_equal']['right_x']}"
                    )
                else:
                    print(f"  ✓ X一致: {checks['x_equal']['left_x']}")

                if not checks["y_opposite"]["result"]:
                    print(
                        f"  ✗ Y非相反数: 左={checks['y_opposite']['left_y']}, 右={checks['y_opposite']['right_y']}"
                    )
                else:
                    print(
                        f"  ✓ Y为相反数: 左={checks['y_opposite']['left_y']}, 右={checks['y_opposite']['right_y']}"
                    )

                if not checks["z_equal"]["result"]:
                    print(
                        f"  ✗ Z不一致: 左={checks['z_equal']['left_z']}, 右={checks['z_equal']['right_z']}"
                    )
                else:
                    print(f"  ✓ Z一致: {checks['z_equal']['left_z']}")

                if comp["all_passed"]:
                    print(f"  → 所有检查通过")
                else:
                    print(f"  → 检查失败")


def check_non_symmetric_links_y_zero(urdf_path: str, tolerance: float = 1e-6) -> Dict:
    """
    检查URDF文件中非左右对称link的惯性一阶矩y分量是否为0

    参数:
        urdf_path: URDF文件路径
        tolerance: y分量判断为0的容差范围

    返回:
        Dict: 包含检查结果的字典
    """

    def parse_xyz(xyz_str: str) -> Optional[List[float]]:
        """解析xyz字符串为浮点数列表"""
        if not xyz_str:
            return None
        try:
            values = [float(x.strip()) for x in xyz_str.split()]
            if len(values) == 3:
                return values
            else:
                return None
        except (ValueError, AttributeError, TypeError):
            return None

    def is_y_zero(y_value: float) -> bool:
        """检查y分量是否为0（考虑容差）"""
        return abs(y_value) <= tolerance

    # 解析URDF文件
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except FileNotFoundError:
        return {"error": f"文件不存在: {urdf_path}"}
    except ET.ParseError as e:
        return {"error": f"XML解析错误: {e}"}
    except Exception as e:
        return {"error": f"解析URDF文件失败: {e}"}

    # 存储结果
    results = {
        "checked_links": [],
        "summary": {
            "total_links": 0,
            "links_with_inertial": 0,
            "non_left_right_links": 0,
            "y_zero_links": 0,
            "y_nonzero_links": 0,
            "no_origin_links": 0,
        },
    }

    # 遍历所有link
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue

        results["summary"]["total_links"] += 1

        # 跳过以left_或right_开头的link
        if link_name.startswith("left_") or link_name.startswith("right_"):
            continue

        # 检查是否有inertial元素
        inertial = link.find("inertial")
        if inertial is None:
            # 没有inertial元素，跳过
            continue

        results["summary"]["links_with_inertial"] += 1
        results["summary"]["non_left_right_links"] += 1

        # 检查是否有origin元素
        origin = inertial.find("origin")
        if origin is None:
            results["summary"]["no_origin_links"] += 1
            link_result = {
                "link_name": link_name,
                "status": "no_origin",
                "xyz": None,
                "y_value": None,
                "message": "没有origin元素",
            }
            results["checked_links"].append(link_result)
            continue

        # 获取xyz属性
        xyz_str = origin.get("xyz")
        xyz = parse_xyz(xyz_str)

        if xyz is None:
            link_result = {
                "link_name": link_name,
                "status": "invalid_xyz",
                "xyz": xyz_str,
                "y_value": None,
                "message": f"无效的xyz值: {xyz_str}",
            }
            results["checked_links"].append(link_result)
            continue

        x, y, z = xyz
        y_zero = is_y_zero(y)

        if y_zero:
            results["summary"]["y_zero_links"] += 1
            status = "y_zero"
            message = f"y={y} (在容差范围内为0)"
        else:
            results["summary"]["y_nonzero_links"] += 1
            status = "y_nonzero"
            message = f"y={y} (不为0)"

        link_result = {
            "link_name": link_name,
            "status": status,
            "xyz": xyz,
            "y_value": y,
            "message": message,
            "tolerance": tolerance,
        }

        results["checked_links"].append(link_result)

    # 计算通过率
    total_checked = results["summary"]["non_left_right_links"]
    if total_checked > 0:
        results["summary"]["compliance_rate"] = (
            results["summary"]["y_zero_links"] / total_checked * 100
        )
    else:
        results["summary"]["compliance_rate"] = 0.0

    return results


def print_y_zero_check_results(results: Dict, verbose: bool = False):
    """
    打印y分量检查结果

    参数:
        results: check_non_symmetric_links_y_zero函数的返回结果
        verbose: 是否显示所有link的详细信息
    """

    if "error" in results:
        print(f"错误: {results['error']}")
        return

    summary = results["summary"]
    checked_links = results["checked_links"]

    print("=" * 70)
    print("URDF非左右对称Link惯性一阶矩Y分量检查结果")
    print("=" * 70)

    print(f"\n统计信息:")
    print(f"  - 总link数: {summary['total_links']}")
    print(f"  - 含有inertial的link数: {summary['links_with_inertial']}")
    print(f"  - 非左右对称link数: {summary['non_left_right_links']}")
    print(f"  - Y=0的link数: {summary['y_zero_links']}")
    print(f"  - Y≠0的link数: {summary['y_nonzero_links']}")
    print(f"  - 无origin元素的link数: {summary['no_origin_links']}")

    if summary["non_left_right_links"] > 0:
        print(f"  - 合规率: {summary['compliance_rate']:.1f}%")

    print(f"\n检查详情:")

    # 按状态分类显示
    status_groups = {}
    for link in checked_links:
        status = link["status"]
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(link)

    # 显示Y≠0的link
    if "y_nonzero" in status_groups:
        print(f"\n  Y≠0的link ({len(status_groups['y_nonzero'])}个):")
        for link in status_groups["y_nonzero"]:
            xyz_str = (
                f"[{link['xyz'][0]:.4f}, {link['xyz'][1]:.4f}, {link['xyz'][2]:.4f}]"
            )
            print(f"    - {link['link_name']}: {xyz_str}")

    # 显示Y=0的link
    if "y_zero" in status_groups and verbose:
        print(f"\n  Y=0的link ({len(status_groups['y_zero'])}个):")
        for link in status_groups["y_zero"]:
            xyz_str = (
                f"[{link['xyz'][0]:.4f}, {link['xyz'][1]:.4f}, {link['xyz'][2]:.4f}]"
            )
            print(f"    - {link['link_name']}: {xyz_str}")

    # 显示其他状态
    for status in status_groups:
        if status not in ["y_zero", "y_nonzero"]:
            print(f"\n  {status}的link ({len(status_groups[status])}个):")
            for link in status_groups[status]:
                print(f"    - {link['link_name']}: {link['message']}")

    # 显示汇总建议
    if summary["y_nonzero_links"] > 0:
        print(f"\n建议:")
        print(f"  以下{summary['y_nonzero_links']}个非对称link的Y分量不为0:")
        for link in checked_links:
            if link["status"] == "y_nonzero":
                print(f"    - {link['link_name']}: y={link['y_value']:.6f}")

        print(f"\n  这可能表示:")
        print(f"    1. 非对称设计的link（正确）")
        print(f"    2. 建模时坐标系未居中（需检查）")
        print(f"    3. 惯性参数估计不准确（需验证）")

    print(
        f"\n容差设置: {summary.get('tolerance', results['checked_links'][0]['tolerance'] if results['checked_links'] else 1e-6)}"
    )


# 使用示例
if __name__ == "__main__":
    # 示例URDF文件路径，请替换为实际路径
    urdf_file = "../urdf/fa_robot.urdf"

    # 执行比较
    results = compare_left_right_inertial_xyz(urdf_file, 0.001)

    # 打印结果（显示详细信息）
    print_comparison_results(results, verbose=False)

    results = check_non_symmetric_links_y_zero(urdf_file, 0.001)
    print_y_zero_check_results(results)

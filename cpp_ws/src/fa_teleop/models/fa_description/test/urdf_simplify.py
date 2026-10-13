import trimesh
import os 


import os

# 给定目录
d = "../"

# 构建完整路径
meshes_path = os.path.join(d, 'meshes')
meshes_origin_path = os.path.join(d, 'meshes_origin')
new_meshes_path = os.path.join(d, 'meshes')

try:
    # 检查原文件夹是否存在
    if os.path.exists(meshes_path) and os.path.isdir(meshes_path):
        # 重命名文件夹
        os.rename(meshes_path, meshes_origin_path)
        print(f"成功将 '{meshes_path}' 重命名为 '{meshes_origin_path}'")
    else:
        print(f"警告: '{meshes_path}' 文件夹不存在")
    
    # 创建新文件夹
    os.makedirs(new_meshes_path, exist_ok=True)
    print(f"成功创建新文件夹 '{new_meshes_path}'")
    
except PermissionError:
    print("错误: 没有足够的权限执行此操作")
except Exception as e:
    print(f"发生错误: {e}")

directory=meshes_origin_path+'/' 
directory_new=new_meshes_path+'/'
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        mesh = trimesh.load_mesh(directory+filename)
        if len(mesh.faces) > 20000:
            simplified = mesh.simplify_quadric_decimation(face_count=20000)
            print(f"  简化到: {len(simplified.faces)} 个面")
            
            # 保存（覆盖原文件或新建）
            output_path = os.path.join(directory, f"simplified_{filename}")
            
            # 保存简化后的网格
            simplified.export(directory_new+filename)
            print(f"  保存到: {output_path}")
        else:
            print(f"  面数已足够少，跳过")


print('done~')

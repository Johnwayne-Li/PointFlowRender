
import numpy as np
from plyfile import PlyData, PlyElement
import pandas as pd

file_dir = r'D:\pyproject\PointCLIP-main\data\1\rotate_3.ply'  # 文件的路径
plydata = PlyData.read(file_dir)  # 读取文件
data = plydata.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
pcl = np.zeros(data_pd.shape, dtype=float)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    pcl[:, i] = data_pd[name]
def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=True)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1200"/>
            <integer name="height" value="900"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""
#点云大小
xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.012"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm


    return [vec[0], vec[1], vec[2]]
xml_segments = [xml_head]

# pcl = np.load(r'D:\pyproject\PointCLIP-main\data\scanobjectnn-C-ply\jitter_4\point_cloud_10.ply')
pcl = standardize_bbox(pcl, 2048)
pcl = pcl[:,[2,0,1]]
pcl[:,0] *= -1
pcl[:,2] += 0.0125

for i in range(pcl.shape[0]):
    color = colormap(pcl[i,0]+0.75,pcl[i,1]+0.75,pcl[i,2]+0.75-0.0125)
    # color = colormap(pcl[i, 0] + 1, pcl[i, 1] + 1, pcl[i, 2] + 1 - 0.0125)
    xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
xml_segments.append(xml_tail)

xml_content = str.join('', xml_segments)

with open('mitsuba_scene.xml', 'w') as f:
    f.write(xml_content)
# def colormap(x, y, z):
#     vec = np.array([x, y, z])
#     vec = np.clip(vec, 0.001, 1.0)
#     norm = np.sqrt(np.sum(vec ** 2))
#     vec /= norm
#     return [vec[0], vec[1], vec[2]]


# 添加一个新的颜色映射，用于蓝色渐变
# def colormap_blue_gradient(x, y, z):
#     blue_min = 0.0
#     blue_max = 1.0
#     blue_step = (blue_max - blue_min) / 255
#
#     blue = np.clip(x * blue_step, blue_min, blue_max)
#     return [blue, blue, 1.0 - blue]
# xml_segments = [xml_head]
#
# # pcl = np.load(r'D:\pyproject\PointCLIP-main\data\scanobjectnn-C-ply\jitter_4\point_cloud_10.ply')
# pcl = standardize_bbox(pcl, 2048)
# pcl = pcl[:,[2,0,1]]
# pcl[:,0] *= -1
# pcl[:,2] += 0.0125
#
# # 在遍历点云时使用新的颜色映射
# for i in range(pcl.shape[0]):
#     color = colormap_blue_gradient(pcl[i, 0], pcl[i, 1], pcl[i, 2])
#     xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
# xml_segments.append(xml_tail)
#
# xml_content = str.join('', xml_segments)
#
# with open('mitsuba_scene.xml', 'w') as f:
#     f.write(xml_content)


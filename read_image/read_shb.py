# python 读取矢量文件
#导入包

from osgeo import ogr

#打开文件（False - read only, True - read/write）

filename = "文件名.shp"
ds = ogr.Open(filename, False)

#获取第一个图层

layer = ds.GetLayer(0)

#获取投影信息

spatialref = layer.GetSpatialRef()

s=spatialref.ExportToWkt()

#图层定义信息

lydefn = layer.GetLayerDefn()

#几何对象类型（点、线、面）

geomtype = lydefn.GetGeomType()

#获取第一个属性字段，字段名、字段类型等

fd0=lydefn.GetFieldDefn(0)

fd0.GetName()
fd0.GetType()
fd0.GetWidth()

#读取数据（空间几何信息及属性信息）

feature=layer.GetNextFeature()

#拿出几何图形

geom=feature.GetGeometryRef()

#查看数据(Wkt给人看，Wkb给计算机看)
geom.ExportToWkt()

#查看空间某一字段信息

feature.GetField('FIPS_CNTRY')

del layer

坚持就是胜利、欧耶~
# Halcon

## 基础

- `xld` Extended Line Descriptions  亚像素轮廓
  - halcon中的Contours 是亚像素数据


## 案例一：图像校正
1. 读取图像
2. 阈值分隔
3. 找边，找顶点坐标
4. 仿射变换矫正
   
``` C
dev_close_window()

read_image(Image_display,'D:/halcon_resources/screen.jpg')

rgb1_to_grapythony (Image_display, GrayImage)

get_image_size(Image_display, imageWidth, imageHeight)

dev_open_window(0, 0, imageWidth, imageHeight, 'black', WindowHandle)

dev_display(GrayImage)

XCoordCorners := []
YCoordCorners := []
threshold(GrayImage, DarkRegion, 0, 180)

connection (DarkRegion, ConnectedRegions)

select_shape_std (ConnectedRegions, SelectedRegions, 'max_area', 70)

reduce_domain (GrayImage, SelectedRegions, ImageReduced)

gen_contour_region_xld (SelectedRegions, Contours, 'border')

segment_contours_xld (Contours, ContoursSplit, 'lines', 5, 4, 2)

count_obj (ContoursSplit, Number)

for index:=1 to Number by 1
    select_obj (ContoursSplit, ObjectSelected, index)
    fit_line_contour_xld (ObjectSelected, 'tukey', -1, 0, 5, 2, RowBegin, ColBegin, RowEnd, ColEnd, Nr, Nc, Dist)
    tuple_concat (XCoordCorners, RowBegin, XCoordCorners)
    tuple_concat (YCoordCorners, ColBegin, YCoordCorners)
endfor

Xoff := 100
Yoff := 200 * imageHeight / imageWidth
hom_vector_to_proj_hom_mat2d (XCoordCorners, YCoordCorners, \
                              [1,1,1,1], [Yoff, Yoff, imageHeight-Yoff, imageHeight-Yoff], \
                              [Xoff, imageWidth-Xoff, imageWidth-Xoff, Xoff], \
                              [1,1,1,1], 'normalized_dlt', HomMat2D)

projective_trans_image (Image_display, TransImage, HomMat2D, 'bilinear', 'false', 'false')

dev_display(TransImage)
```
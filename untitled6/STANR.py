import gdal
import sys
import numpy as np
import os
gdal.UseExceptions()
image_list = []
processed_image = {}
np.seterr(all='raise')
def hist_match(input_image, reference_image): #1번 파일을 변형, 따라서 리턴 값과 2번 파일을 사용해야함

    oldshape = input_image.shape
    source = reference_image.ravel()
    template = input_image.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    matched_image = interp_t_values[bin_idx].reshape(oldshape)

    return matched_image
def load_image(name_list):
    for name in name_list:
        try:
            image_list.append(gdal.Open(name))
        except:
            print("Can't not open "+name+" file...")
            sys.exit(1)
    return
def heterogeneity(img,x_pos,y_pos,win_size):
    subset_img= img[max(0,x_pos-win_size):min(image_list[0].RasterXSize,x_pos+win_size+1),max(0,y_pos-win_size):min(image_list[0].RasterYSize,y_pos+win_size+1)]
    subset_img_serial= subset_img.ravel()
    std =np.std(subset_img_serial)
    mean = np.mean(subset_img_serial)
    ##디버깅용
    #if mean ==0.:
    #    print(x_pos,y_pos,win_size)
    #    print(subset_img)
    #    print("평균",mean)
    #    print("표준 편차",std)
        #print("표준 편차/평균",std/mean)
    #디버그 종료
    try:
        return std/mean
    except:
        #print("분모 0 케이스 발생",mean)
        return 0.
def spatial_element(img,x_pos,y_pos,min_size=5,max_size=11,delta_t=0.05): #min_size = minimum window size, max_size = maximum window size, delta_t = threshhold of heterogeneity
    for win_size in range(max_size,min_size-2,-2):
        s=heterogeneity(img,x_pos,y_pos,(win_size-1)//2)
        if delta_t> s:#heterogeneity(img,x_pos,y_pos,(win_size-1)//2):
            return win_size
        #디버깅용
        #a= input()
        #디버깅 종료
    return min_size
def spatial_table(img,x_range,y_range): #return spatial adaptive window table N(t)
    spatial_table=[]
    for x_pos in range(0,x_range):
        spatial_table.append([])
        for y_pos in range(0,y_range):
            spatial_table[x_pos].append(spatial_element(img,x_pos,y_pos))
    return np.asarray(spatial_table,dtype=np.uint32)
def ncenter_heterogeneity(img,x_pos,y_pos,win_size): #검사 필요 함수
    subset_img = img[max(0,x_pos-win_size):min(image_list[0].RasterXSize,x_pos+win_size+1),
                max(0,y_pos-win_size):min(image_list[0].RasterYSize,y_pos+win_size+1)]
    min_x=max(0,x_pos-win_size)
    min_y=max(0,y_pos-win_size)
    #print(subset_img)
    #print(x_pos,y_pos,win_size)
    subset_img_serial= subset_img.ravel()
    #print(x_pos,y_pos,min_x,min_y,subset_img.shape[1])
    subset_img_serial=np.delete(subset_img_serial,x_pos-min_x+subset_img.shape[1]*(y_pos-min_y))
    #print(subset_img_serial)
    std =np.std(subset_img_serial)
    mean = np.mean(subset_img_serial)
    return std/mean
def ncenter_spatial_element(img,win_size,x_pos,y_pos): #min_size = minimum window size, max_size = maximum window size, delta_t = threshhold of heterogeneity
    s = ncenter_heterogeneity(img, x_pos, y_pos, int((win_size - 1) // 2))
    return s
def ncenter_spatial_table(img,window_table,x_range,y_range): #return spatial adaptive window table N(t)
    spatial_table=[]
    for x_pos in range(0,x_range):
        #print(x_pos)
        spatial_table.append([])
        for y_pos in range(0,y_range):
            spatial_table[x_pos].append(ncenter_spatial_element(img,window_table[x_pos][y_pos],x_pos,y_pos))
    spatial_table=np.asarray(spatial_table,dtype=np.float64)#여기서 노말라이즈를 진행함
    spatial_table=spatial_table/spatial_table.max()
    return spatial_table
def ncenter_mean(img,x_pos,y_pos,win_size): #검사 필요 함수
    subset_img= img[max(0,x_pos-win_size):min(image_list[0].RasterXSize,x_pos+win_size+1),max(0,y_pos-win_size):min(image_list[0].RasterYSize,y_pos+win_size+1)]
    min_x = max(0, x_pos - win_size)
    min_y = max(0, y_pos - win_size)
    #print(subset_img)
    #print(x_pos, y_pos, win_size)
    subset_img_serial = subset_img.ravel()
    #print(x_pos, y_pos, min_x, min_y, subset_img.shape[1])
    subset_img_serial = np.delete(subset_img_serial, x_pos - min_x + subset_img.shape[1] * (y_pos - min_y))
    #print(subset_img_serial)
    mean = np.mean(subset_img_serial)
    return mean
def ncenter_mean_element(img,win_size,x_pos,y_pos,): #min_size = minimum window size, max_size = maximum window size, delta_t = threshhold of heterogeneity
    s = ncenter_mean(img, x_pos, y_pos, int((win_size - 1) // 2))
    return s
def ncenter_mean_table(img,window_table,x_range,y_range): #return spatial adaptive window table N(t)
    mean_table=[]
    for x_pos in range(0,x_range):
        mean_table.append([])
        for y_pos in range(0,y_range):
            mean_table[x_pos].append(ncenter_mean_element(img,window_table[x_pos][y_pos],x_pos,y_pos))
    return np.asarray(mean_table,dtype=np.float64)
def dinr_element(img1,img2, x_pos, y_pos,img_nh_map1,img_nh_map2,img_mean_map1,img_mean_map2):
    nh1=img_nh_map1[x_pos][y_pos]
    nh2=img_nh_map2[x_pos][y_pos]
    mean1=img_mean_map1[x_pos][y_pos]
    mean2=img_mean_map2[x_pos][y_pos]
    child=min(img1[x_pos][y_pos]*nh1+(1-nh1)*mean1,img2[x_pos][y_pos]*nh2+(1-nh1)*mean2)
    parent=max(img1[x_pos][y_pos]*nh1+(1-nh1)*mean1,img2[x_pos][y_pos]*nh2+(1-nh1)*mean2)
    try:
        result=1-(child/parent)
    except:
        print(child,parent)
        result=0
    return result
def dinr_table(img1,img2,x_range,y_range,img_nh_map1,img_nh_map2,img_mean_map1,img_mean_map2):
    dinr_table = []
    for x_pos in range(0, x_range):
        dinr_table.append([])
        for y_pos in range(0, y_range):
            dinr_table[x_pos].append(dinr_element(img1,img2, x_pos, y_pos,img_nh_map1,img_nh_map2,img_mean_map1,img_mean_map2))
    return np.asarray(dinr_table, dtype=np.float64)
def save_image(eType,band_c=None): # using processed_image {name=value} saving the files
                        # using dictionay value는 3중 배열, 테이블의 배열 형태이어야 한다
    driver = image_list[0].GetDriver()
    if band_c is None:
        band_count =image_list[0].RasterCount
    else:
        band_count =band_c
    for name,table in processed_image.items():
        try:
            ds = driver.Create(name, xsize=image_list[0].RasterXSize, ysize=image_list[0].RasterYSize, bands=band_count,
                               eType=eType, options=["COMPRESS=LZW"])
        except:
            print("Create "+name+" file failed...")
            sys.exit(1)
        for band_num in range(0, band_count):
            ds_band = ds.GetRasterBand(band_num + 1)
            if eType==gdal.GDT_UInt32:
                ds_band.WriteArray(table[band_num].astype('uint32'))
            elif eType==gdal.GDT_UInt16:
                ds_band.WriteArray(table[band_num].astype('uint16'))
            elif eType==gdal.GDT_Byte:
                ds_band.WriteArray(table[band_num].astype('uint8'))
            else:
                ds_band.WriteArray(table[band_num].astype('float64'))
if __name__ == '__main__':
    getBandArray = lambda band,dataset: dataset.GetRasterBand(band+1).ReadAsArray()
    src1 ="test1.tif"
    src2 ="test2.tif"
    load_image([src1,src2])
    band_count = image_list[0].RasterCount # 이 변수에 밴드 갯수 저장
    image_Xsize=image_list[0].RasterXSize
    image_Ysize=image_list[0].RasterYSize

    if image_list[0].ReadAsArray().dtype==np.uint16 and image_list[1].ReadAsArray().dtype==np.uint16 :
        processed_image["processed_" + src1] = []
        processed_image["processed_" + src2] = []
        for band_num in range(0,band_count):
            processed_image["processed_" + src1].append(getBandArray(band_num, image_list[0]))
            processed_image["processed_" + src2].append(getBandArray(band_num, image_list[1]))
        save_image(gdal.GDT_UInt16,1)
        processed_image.clear()
    src1 = "processed_" + src1
    src2 = "processed_" + src2
    image_list.clear()
    load_image([src1, src2])
    band_count = image_list[0].RasterCount
    if os.path.exists('filtered.tif')==False: #filtered.tif파일이 존재하지 않으면
        processed_image['filtered.tif']=[]
        for band_num in range(0,band_count):
            processed_image['filtered.tif'].append(
                hist_match(getBandArray(band_num,image_list[0]),getBandArray(band_num,image_list[1]))) # image_list[0]과 filtered.tif 사용하세요
        save_image(gdal.GDT_UInt16)
        processed_image.clear()
        src2="filtered.tif"
    #이미지를 필터된 이미지로 교체 #이게 더 보기 좋은 듯
    image_list.clear()
    load_image([src1,src2])
    #N_T1파일 존재 확인..
    if os.path.exists('N_T1.tif')==False:
        processed_image['N_T1.tif'] = []
        for band_num in range(0, band_count):
            processed_image['N_T1.tif'].append(
                spatial_table(getBandArray(band_num,image_list[0]), image_Xsize, image_Ysize))
    #N_T2파일 존재 확인..
    if os.path.exists('N_T2.tif')==False:
        processed_image['N_T2.tif'] = []
        for band_num in range(0, band_count):
            processed_image['N_T2.tif'].append(
                spatial_table(getBandArray(band_num,image_list[1]), image_Xsize, image_Ysize))
    #처리한 N_T1,N_T2 데이터를 저장...
    if processed_image!={}:
        save_image(gdal.GDT_Float64)
        processed_image.clear()
    #저장한 N_T1과 N_T2를 각각 image_list[2]와 image_list[3]에 로드
    load_image(["N_T1.tif", "N_T2.tif"])
    #DINR을 구하기 위한 전처리 작업 시작
    #의문 d_max는 어디에서의 최대를 뜻하는 거지?
    #최대값을 구하기 위해 heterogenity 테이블을 만들어야 한다.(center pixel이 제거된 영역으로 구한 값 중 최대값으로 생각하겠다.)
    #img_nh_map1파일 존재 확인...
    if os.path.exists('img_nh_map1.tif')==False:
        processed_image['img_nh_map1.tif'] = []
        for band_num in range(0, band_count):
            processed_image['img_nh_map1.tif'].append(
                ncenter_spatial_table(
                    getBandArray(band_num,image_list[0]),#image 1번
                    getBandArray(band_num, image_list[2]),#N_T1
                    image_Xsize, image_Ysize
                )
            )
            #print("one loop done~!!")
    # img_nh_map2파일 존재 확인...
    if os.path.exists('img_nh_map2.tif') == False:
        processed_image['img_nh_map2.tif'] = []
        for band_num in range(0, band_count):
            processed_image['img_nh_map2.tif'].append(
                ncenter_spatial_table(
                    getBandArray(band_num, image_list[1]),#image 2번
                    getBandArray(band_num, image_list[3]),#N_T2
                    image_Xsize, image_Ysize
                )
            )
    #처리한 normalize hetro 데이터를 저장한다.
    if processed_image!={}:
        save_image(gdal.GDT_Float64)
        processed_image.clear()

    # img_mean_map1파일 존재 확인...
    if os.path.exists('img_mean_map1.tif') == False:
        processed_image['img_mean_map1.tif'] = []
        for band_num in range(0, band_count):
            processed_image['img_mean_map1.tif'].append(
                ncenter_mean_table(
                    getBandArray(band_num, image_list[0]),  # image 1번
                    getBandArray(band_num, image_list[2]),  # N_T1
                    image_Xsize, image_Ysize
                )
            )
    # img_mean_map2파일 존재 확인...
    if os.path.exists('img_mean_map2.tif') == False:
        processed_image['img_mean_map2.tif'] = []
        for band_num in range(0, band_count):
            processed_image['img_mean_map2.tif'].append(
                ncenter_mean_table(
                    getBandArray(band_num, image_list[1]),  # image 2번
                    getBandArray(band_num, image_list[3]),  # N_T2
                    image_Xsize, image_Ysize
                )
            )
    # 처리한 normalize hetro 데이터를 저장한다.
    if processed_image != {}:
        save_image(gdal.GDT_Float64)
        processed_image.clear()
    #진짜 DINR 작업 시작
    #처리에 필요한 이미지를 모두 로드한다.
    image_list.clear()
    load_image([src1,src2, "img_nh_map1.tif", "img_nh_map2.tif","img_mean_map1.tif","img_mean_map2.tif"])
    processed_image['DINR.tif'] = []
    for band_num in range(0, band_count):
        processed_image['DINR.tif'].append(
            dinr_table(
                getBandArray(band_num, image_list[0]),#첫번째 이미지
                getBandArray(band_num, image_list[1]),#두번째 이미지
                image_Xsize, image_Ysize,
                getBandArray(band_num, image_list[2]),#첫번째 이미지의 normalized hetero
                getBandArray(band_num, image_list[3]),#두번째 이미지의 normalized hetero
                getBandArray(band_num, image_list[4]),#첫번째 이미지의 평균
                getBandArray(band_num, image_list[5]))#첫번째 이미지의 평균
        )
    save_image(gdal.GDT_Float64)
    processed_image.clear()
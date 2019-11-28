import gdal
import sys
import numpy as np
gdal.UseExceptions()
def hist_match(input_image, reference_image):

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
def Usage():
    print("""
    $ getrasterband.py [ band number ] input-raster
    """)
    sys.exit(1)

def main( band_num, input_file1 ,input_file2):
    window=1
    ###################################### INPUT: src_ds OUTPUT: dst_ds  ######################################
    try:
        src_ds1 = gdal.Open( input_file1 ) #Open source file1
        src_ds2 = gdal.Open( input_file2 ) #Open source file2

        #r_file = gdal.Open("Rmag.tif")  # Open source file2 비교용 파일 오픈
    except:
        print('Can not open file...')
        sys.exit(1)
    driver = src_ds1.GetDriver()
    x_size= src_ds1.RasterXSize
    y_size = src_ds1.RasterYSize
    #여기
    try:
        dsmin1 = driver.Create("PyDsmin1_"+str(window*2+1)+".tif",xsize=x_size, ysize=y_size,bands=src_ds1.RasterCount, eType=gdal.GDT_Float32,options=["COMPRESS=LZW"]) #Make save file
        dsmin2 = driver.Create("PyDsmin2_"+str(window*2+1)+".tif",xsize=x_size, ysize=y_size,bands=src_ds1.RasterCount, eType=gdal.GDT_Float32,options=["COMPRESS=LZW"])
        difall = driver.Create("PyDifall_"+str(window*2+1)+".tif",xsize=x_size, ysize=y_size,bands=src_ds1.RasterCount, eType=gdal.GDT_Float32,options=["COMPRESS=LZW"])
        filtered = driver.CreateCopy("filtered.tif",src_ds1)
    except:
        print('Can not make result file...')
        sys.exit(1)
    before_stack_match = []
    for band_num in range(0, src_ds1.RasterCount):
        src_arr1 = src_ds1.GetRasterBand(band_num + 1).ReadAsArray()
        src_arr2 = src_ds2.GetRasterBand(band_num + 1).ReadAsArray()
        before_match = hist_match(src_arr1, src_arr2)
        before_stack_match.append(before_match)
    src_before = np.asarray(before_stack_match).astype('uint16')
    for band_num in range(0, src_ds1.RasterCount):
        filtered.GetRasterBand(band_num + 1).WriteArray(src_before[band_num])
    ############################################  PROCESSS START  #############################################\
    XSize = src_ds1.GetRasterBand(1).XSize
    YSize = src_ds1.GetRasterBand(1).YSize

    for band_num in range(0, src_ds1.RasterCount):
        ### 비교용 파일
        #r_band= r_file.GetRasterBand(band_num)
        #r_arr =r_band.ReadAsArray()
        ###
        srcband1 = src_ds1.GetRasterBand(band_num+1)
        dsmin_band1 = dsmin1.GetRasterBand(band_num+1)
        dsmin_band2 = dsmin2.GetRasterBand(band_num+1)
        difall_band = difall.GetRasterBand(band_num+1)
        dsmin_arr1 = dsmin_band1.ReadAsArray().astype('float32')
        dsmin_arr2 = dsmin_band2.ReadAsArray().astype('float32')
        difall_arr = difall_band.ReadAsArray().astype('float32')
        src_arr1 = src_before[band_num].astype('float32')
        src_arr2 = srcband1.ReadAsArray().astype('float32')
        for i in range(0,XSize):
            for j in range(0,YSize):
                x_low = max(0,i - window)
                x_high = min(i + window,XSize-1)+1
                y_low = max(0, j - window)
                y_high = min(j + window, YSize - 1)+1
                dsmin_iter1=src_arr1[x_low:x_high,y_low:y_high].reshape(1,(x_high-x_low)*(y_high-y_low))[0]
                dsmin_iter2=src_arr2[x_low:x_high,y_low:y_high].reshape(1,(x_high-x_low)*(y_high-y_low))[0]
                x_diff_a = max(src_arr2[i][j] - np.amax(dsmin_iter1),0)
                x_diff_b = max(src_arr1[i][j] - np.amax(dsmin_iter2),0)
                dsmin_arr1[i][j]=x_diff_a
                dsmin_arr2[i][j]=x_diff_b
                ################################
                if x_diff_a ==0:
                    difall_arr[i][j] = 0 - x_diff_b
                else:
                    difall_arr[i][j] = x_diff_a

        difall_band.WriteArray(difall_arr)
        dsmin_band1.WriteArray(dsmin_arr1)
        dsmin_band2.WriteArray(dsmin_arr2)
        test1= dsmin_band1.ReadAsArray()
    #####################################  X_diff image making is done   #######################################
    ####여기
    '''
    difall = gdal.Open("PyDifall.tif")  # Open source file2 비교용 파일 오픈
    r_band = r_file.GetRasterBand(1)
    r_arr = r_band.ReadAsArray()
    XSize = src_ds1.GetRasterBand(1).XSize
    YSize = src_ds1.GetRasterBand(1).YSize
    window = 1
    '''
    ####
    x_diff_list = []
    mag = driver.Create("Pymag_"+str(window*2+1)+".tif", xsize=x_size, ysize=y_size,bands=1, eType=gdal.GDT_Float32,options=["COMPRESS=LZW"])
    dir = driver.Create("Pydir_"+str(window*2+1)+".tif", xsize=x_size, ysize=y_size,bands=1, eType=gdal.GDT_Float32,options=["COMPRESS=LZW"])
    mag_arr = mag.GetRasterBand(1).ReadAsArray()
    dir_arr = dir.GetRasterBand(1).ReadAsArray()
    for band_num in range(src_ds1.RasterCount):
        band_num += 1
        x_diff_list.append(difall.GetRasterBand(band_num).ReadAsArray())
    for i in range(0, XSize):
        for j in range(0, YSize):
            sum = np.float32(0)
            square_sum = np.float64(0)
            for x_diff in x_diff_list:
                sum = np.add((x_diff[i][j]),sum)
                aa = np.power(x_diff[i][j],2)
                square_sum = np.add(np.power(x_diff[i][j],2),square_sum)
            magnitude = np.sqrt(square_sum)
            if magnitude != 0:
                direction = 180/np.pi*np.arccos(1/np.sqrt(src_ds1.RasterCount)*(sum/magnitude))
            else:
                direction = 0

            mag_arr[i][j]=magnitude
            dir_arr[i][j]=direction
    mag.GetRasterBand(1).WriteArray(mag_arr)
    dir.GetRasterBand(1).WriteArray(dir_arr)
if __name__ == '__main__':
    main( 4, 'test2.tif','test1.tif' )
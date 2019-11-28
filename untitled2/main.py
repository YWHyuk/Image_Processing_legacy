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
def dL1(h_a,h_b):
    total=0
    h_a=np.subtract(h_a,h_b)
    h_a=np.abs(h_a)
    return np.sum(h_a)
def dCan(h_a,h_b):
    total=0
    h_c=np.subtract(h_a,h_b)
    h_c=np.abs(h_c)
    h_d=np.add(h_a,h_b)

    temp= h_d.size
    for i in range(temp-1,-1,-1):
        if h_d[i]!=0:
            h_c=h_c[0:i+1]
            h_d=h_d[0:i+1]
            break;
    h_c=np.divide(h_c,h_d)
    h_c=np.sum(h_c)
    return h_c
def Usage():
    sys.exit(1)

def main( band_num, input_file1 ,input_file2):
    ###################################### INPUT: src_ds OUTPUT: dst_ds  ######################################
    try:
        src_ds1 = gdal.Open( input_file1 ) #Open source file1
        src_ds2 = gdal.Open( input_file2 ) #Open source file2
    except:
        print('Can not open file...')
        sys.exit(1)
    driver = src_ds1.GetDriver()
    #여기
    for i in range(2,4,2):
        window = i
        try:
            dsmin1 = driver.Create("new_src\\PyCan_"+str((window*2)+1)+".tif",xsize=src_ds1.RasterXSize, ysize=src_ds1.RasterYSize,bands=src_ds1.RasterCount, eType=gdal.GDT_Float32,options=["COMPRESS=LZW"]) #Make save file
            dsmin2 = driver.Create("new_src\\PyDl_" + str((window * 2) + 1) + ".tif", xsize=src_ds1.RasterXSize, ysize=src_ds1.RasterYSize,
                                   bands=src_ds1.RasterCount, eType=gdal.GDT_Float32,
                                   options=["COMPRESS=LZW"])  # Make save file
            #test_8_1 = driver.Create("8_test1.tif", xsize=600, ysize=600, bands=1, eType=gdal.GDT_UInt16,options=["COMPRESS=LZW"])
            #test_8_2 = driver.Create("8_test2.tif", xsize=600, ysize=600, bands=1, eType=gdal.GDT_UInt16,options=["COMPRESS=LZW"])
        except:
            print('Can not make result file...')
            sys.exit(1)
        before_stack_match = []
        for band_num in range(0,src_ds1.RasterCount):
            src_arr1 =  src_ds1.GetRasterBand(band_num+1).ReadAsArray()
            src_arr2 =  src_ds2.GetRasterBand(band_num+1).ReadAsArray()
            before_match=hist_match(src_arr1, src_arr2)
            before_stack_match.append(before_match)
        src_before = np.asarray(before_stack_match).astype('uint16')
        ############################################  PROCESSS START  #############################################\
        XSize = src_ds1.GetRasterBand(1).XSize
        YSize = src_ds1.GetRasterBand(1).YSize
        for band_num in range(0,1):#src_ds1.RasterCount):
            dsmin_arr1 = dsmin1.GetRasterBand(band_num+1).ReadAsArray()
            dsmin_arr2 = dsmin2.GetRasterBand(band_num + 1).ReadAsArray()
            src_arr1 = src_before[band_num].astype('int32')
            src_arr2 = src_ds1.GetRasterBand(band_num+1).ReadAsArray().astype('int32')
            #np.divmod(src_arr1,np.uint16(2**8),src_arr1)
            #np.divmod(src_arr2,np.uint16(2**8),src_arr2)
            for i in range(0,XSize):
                print(i)
                for j in range(0,YSize):
                    x_low = max(0,i - window)
                    x_high = min(i + window,XSize-1)+1
                    y_low = max(0, j - window)
                    y_high = min(j + window, YSize - 1)+1
                    dsmin_iter1=src_arr1[x_low:x_high,y_low:y_high].ravel().astype('uint8')
                    dsmin_iter2=src_arr2[x_low:x_high,y_low:y_high].ravel().astype('uint8')
                    #dsmin_iter1 = np.sort(dsmin_iter1)
                    #dsmin_iter2 = np.sort(dsmin_iter2)
                    aa= np.histogram(dsmin_iter1,bins=2**8,range=(0,2**8),density=True)
                    bb=np.histogram(dsmin_iter2,bins=2**8,range=(0,2**8),density=True)
                    aa[0][::-1].sort()
                    aa=aa[0]
                    bb[0][::-1].sort()
                    bb=bb[0]
                    x_diff_a =dCan(aa,bb)
                    x_diff_b =dL1(aa,bb)
                    dsmin_arr1[i][j]=x_diff_a.astype('float32')
                    dsmin_arr2[i][j] = x_diff_b.astype('float32')
            dsmin_band1=dsmin1.GetRasterBand(band_num+1)
            dsmin_band1.WriteArray(dsmin_arr1)
            dsmin_band2 = dsmin2.GetRasterBand(band_num + 1)
            dsmin_band2.WriteArray(dsmin_arr2)
                ################################
    #####################################  X_diff image making is done   #######################################

if __name__ == '__main__':
    #main(1, 'debug1.tif', 'debug2.tif')
    main( 1, '2014_set1_worldview_pan.tif','2015_set1_worldview_pan.tif' )
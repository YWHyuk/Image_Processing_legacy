import gdal
import sys
import numpy as np
img_list=[];
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
def norm_diff(img1,img2,mod): #img* should be tif file
    if mod=="NDR":
        target_1=1
        target_2=1
    elif mod=="NDVI":
        target_1=3
        target_2=4
    arr1 = img1.GetRasterBand(target_1).ReadAsArray().astype('float64')
    arr2 = img2.GetRasterBand(target_2).ReadAsArray().astype('float64')
    temp = min(arr1.min(),arr2.min(),0)
    arr1 = arr1 - temp
    arr2 = arr2 - temp
    np.seterr(divide='print')
    normed_arr = (arr2-arr1)/(arr2+arr1)
    return normed_arr
def image_append(name):
    try:
        img_list.append(gdal.Open(name))
    except:
        print('Can not open file...')
        sys.exit(1)
def save_image(name,driver,arr,band=1,band_num=1):
    ds = driver.Create(name, xsize=400, ysize=400,
                           bands=band, eType=gdal.GDT_Float64
                           )  # Make save file
    ds.GetRasterBand(band_num).WriteArray(arr)
def main(name_list):
    for name in name_list:
        image_append(name)
    ndr = norm_diff(img_list[0],img_list[1],"NDR")

    pyndvi1=norm_diff(img_list[2],img_list[2],"NDVI")
    pyndvi2=norm_diff(img_list[3],img_list[3],"NDVI")

    ndvi_diff= pyndvi2-pyndvi1
    ndr =hist_match(ndvi_diff, ndr)
    yes= np.logical_or(np.logical_and(ndr>=0,ndvi_diff>=0),np.logical_and(ndr<0,ndvi_diff<0))
    no = np.logical_not(yes)
    change_arr = (ndr+ndvi_diff)*yes + (ndr-ndvi_diff)*no

    driver = img_list[2].GetDriver()
    #save_image("py_ndvi2014.tif", driver, pyndvi1)
    #save_image("py_ndvi2016.tif", driver, pyndvi2)
    save_image("plus.tif", driver, yes.astype('float64'),3,2)
    save_image("minus.tif",driver,no.astype('float64'),3,1);
    save_image("ndr.tif", driver, ndr)
    save_image("ndr.tif",driver,ndr)
    save_image("ndvi_diff.tif",driver,ndvi_diff)
    save_image("result.tif",driver,change_arr)
if __name__ == '__main__':
    #main(1, 'debug1.tif', 'debug2.tif')
    main(['aa2014.tif','aa2016.tif','clip_2014_landsat.tif','clip_2016_landsat.tif'])




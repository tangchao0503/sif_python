import numpy as np
from scipy.special import eval_legendre
from scipy import optimize
# 所有提取方法均基于xarray数据，xarray数据的变量与波长、时间、光谱仪绑定
def cal_inside_bands_ave(data):
    '''
    根据多个光谱找出窗口内数据最低点对应波长
    '''
    sky_spec = data.sky
    veg_spec = data.veg
    wvl_inside_band_l = np.mean(veg_spec.idxmin(dim='Wavelength')).values
    wvl_inside_band_e = np.mean(sky_spec.idxmin(dim='Wavelength')).values
    return[wvl_inside_band_l,wvl_inside_band_e]

def cal_outside_values_mean(data,outer):
    '''
    计算肩部窗口的均值
    '''
    _data = data.where((data.Wavelength>outer[0])&(data.Wavelength<outer[1]),drop=True)
    wvl_outer_mean = _data['Wavelength'].mean().values
    _mean = _data.mean(dim = 'Wavelength')
    E_out = _mean.sky.values
    L_out = _mean.veg.values
    return L_out,E_out,wvl_outer_mean

def sfld(data,wl_range,outer):

    """
    Standard FLD (Huaize Feng)

    input:
          data,xarray dataset
          wl_range, a window around Fraunhofer lines position, [start,end]
          outer, a window outside the Fraunhofer lines, [start,end]
    """
    sif = []
    refl = []
    nmeas_ = data.Measures.size
    
    data = data.where((data.Wavelength>wl_range[0])&(data.Wavelength<wl_range[1]),drop=True)
    [wvl_inside_band_l,wvl_inside_band_e]=cal_inside_bands_ave(data)
    for i in range(0,nmeas_):
        _data = data.isel(Measures=i)
        veg_out,sky_out,_ = cal_outside_values_mean(_data,outer)
        veg_in = _data.veg.sel(Wavelength = wvl_inside_band_l,method='nearest').values
        sky_in = _data.sky.sel(Wavelength = wvl_inside_band_e,method='nearest').values
        _sif = (sky_out*veg_in - sky_in*veg_out)/(sky_out - sky_in) 
        _refl = (veg_in - _sif)/ sky_in
        sif.append(_sif)
        refl.append(_refl)
    return[sif,refl]

def fld3(data,wl_range,outer_left,outer_right):
    """
    3FLD (Huaize Feng)
        input:
          data,xarray dataset
          wl_range, a window around Fraunhofer lines position, [start,end]
          outer_*, a window outside the Fraunhofer lines, [start,end]
              outer_left, left window outside the Fraunhofer lines
              outer_right, left window outside the Fraunhofer lines
        return:
          list[sif,reflectance]
    """
    sif = []
    refl = []
    nmeas_ = data.Measures.size
    
    data = data.where((data.Wavelength>wl_range[0])&(data.Wavelength<wl_range[1]),drop=True)
    [wvl_inside_band_l,wvl_inside_band_e]=cal_inside_bands_ave(data)
    for i in range(0,nmeas_):
        _data = data.isel(Measures=i)
        veg_out_left,sky_out_left,wvl_outer_left = cal_outside_values_mean(_data,outer_left)
        veg_out_right,sky_out_right,wvl_outer_right = cal_outside_values_mean(_data,outer_right)
        veg_in = _data.veg.sel(Wavelength = wvl_inside_band_l,method='nearest').values
        sky_in = _data.sky.sel(Wavelength = wvl_inside_band_e,method='nearest').values
        
        # 根据离吸收峰的距离的反比进行赋权
        wight_left = (wvl_outer_right - wvl_inside_band_e)/(wvl_outer_right - wvl_outer_left)
        wight_right = (wvl_inside_band_e - wvl_outer_left)/(wvl_outer_right - wvl_outer_left)
        
        _sif =  (veg_in - (sky_in/((wight_left*sky_out_left) + (wight_right*sky_out_right))) * ((wight_left*veg_out_left) + (wight_right*veg_out_right))) / (1-(sky_in / ((wight_left*sky_out_left) + (wight_right*sky_out_right))))
        _refl = (veg_in - _sif)/ sky_in
        sif.append(_sif)
        refl.append(_refl)
    return[sif,refl]        

def sfm(data,wl_range,band):

    """
    Spectral Fitting Method (Huaize Feng)
    input:
          data,
              xarray dataset
          wl_range,[start,end]
              a window around Fraunhofer lines position, ,about 10 nm
              for fitting a ployniam or gussian function.
          band,float/int
              exact position of the Fraunhofer line
    return:
          list[sif,reflectance,rmse,B]
              sif, numpy array, sif at the Fraunhofer line for each measurement
              reflectance, numpy array
              rmse, RMSE
              B, numpy array, the parameters of the fitting equation
                  [a,b,c,d,e,f]: a, b, c for polynimal refelectance
                                 d, float, MAX sif value [0,10] mw/m2/nm/sr
                                 e, position of MAX sif value - wl_range, [0,wavelength.size], nm
                                 f, full width at half maximum, [0,wavelength.size*5]
    """
    sif,refl,rmse,B = [],[],[],[]
    data = data.where((data.Wavelength>wl_range[0])&(data.Wavelength<wl_range[1]),drop=True)
    abosorb_line_position = np.where(data.Wavelength == data.Wavelength.sel(Wavelength=760,method='nearest').values)
    _nmeas = data.Measures.size
    _nwvl = data.Wavelength.size
    _x = (data.Wavelength -np.min(data.Wavelength)).values
    poly_refl = [_x**2,_x,np.ones(_nwvl)]
    poly_sif = [_x**2,_x,np.ones(_nwvl)]
   
    for i in range(0,_nmeas):
        sky_spec = data.sky[i].values
        veg_spec = data.veg[i].values
        _X = np.concatenate([poly_refl*sky_spec,poly_refl]).T
        _B = np.linalg.lstsq(_X,veg_spec,rcond=-1)
        _sif = np.array(poly_sif).T.dot(_B[0][-3:])
        _refl = (veg_spec - _sif) / (sky_spec+0.000001)
        sif.append(_sif[abosorb_line_position][0])
        refl.append(_refl[abosorb_line_position][0])
        rmse.append(np.sqrt(np.sum(_B[1]**2)/_nwvl))
        B.append(_B[0])
    return [sif,refl,rmse,B]


def f(x, a, b, c, d, e, f):
    sky_spec = x['sky_spec']
    _x = x['_x']
    refl = a * _x**2*sky_spec + b* _x*sky_spec + c*sky_spec
    sif = d*np.exp(-(_x-e)**2/f)
    y_hat = refl + sif
    return y_hat

def f_cal(X, a,b,c,d,e,f):
    sky_spec = X['sky_spec']
    _x = X['_x']
    refl = a * _x**2*sky_spec + b* _x*sky_spec + c*sky_spec
    sif = d*np.exp(-(_x-e)**2/f)
    y_hat = refl + sif
    return sif,refl

def sfm_gaussian(data,wl_range,band=760):
    
    '''
    Spectral Fitting Method (Gaussian Ver. (Huaize Feng))
    input:
          data,
              xarray dataset
          wl_range,[start,end]
              a window around Fraunhofer lines position, ,about 10 nm
              for fitting a polynomial or gaussian function.
          band,float/int
              exact position of the Fraunhofer line
    return:
          list[sif,reflectance,rmse,B]
              sif, numpy array, sif at the Fraunhofer line for each measurement
              reflectance, numpy array
              rmse, RMSE
              B, numpy array, the parameters of the fitting equation
                  [a,b,c,d,e,f]: a, b, c for polynimal refelectance
                                 d, float, MAX sif value [0,10] mw/m2/nm/sr
                                 e, position of MAX sif value - wl_range, [0,wavelength.size], nm
                                 f, full width at half maximum
    '''
    sif,refl,rmse,B = [],[],[],[]
    data = data.where((data.Wavelength>wl_range[0])&(data.Wavelength<wl_range[1]),drop=True)
    abosorb_line_position = np.where(data.Wavelength == data.Wavelength.sel(Wavelength=band,method='nearest').values)
    _nmeas = data.Measures.size
    _nwvl = data.Wavelength.size
    _x = (data.Wavelength -np.min(data.Wavelength)).values
    poly_refl = [_x**2,_x,np.ones(_nwvl)]
    poly_sif = [_x**2,_x,np.ones(_nwvl)]
    
    bounds=((-np.inf, -np.inf,-np.inf, 0, 0, 0), (np.inf, np.inf,np.inf, 10, _nwvl, _nwvl*5))
    
    for i in range(0,_nmeas):
        sky_spec = data.sky[i].values
        veg_spec = data.veg[i].values
        _X = {'sky_spec':sky_spec,'_x':_x}
        _B = optimize.curve_fit(f, _X, veg_spec,bounds=bounds)[0]
#         print(_B)
        
        _sif,_ = f_cal(_X,*_B)
#         print(_sif)
        _rmse = np.sqrt(np.sum((f(_X,*_B)-veg_spec)**2)/_nwvl)
        _refl = (veg_spec - _sif) / sky_spec
        sif.append(_sif[abosorb_line_position][0])
        refl.append(_refl[abosorb_line_position][0])
        rmse.append( _rmse)
        B.append(_B[0])
    return [sif,refl,rmse,B]

def doas(data,wl_range,band=760):

    """
    Spectral Fitting Method (Huaize Feng)
    input:
          data,
              xarray dataset
          wl_range,[start,end]
              a window around Fraunhofer lines position, ,about 10 nm
              for fitting a ployniam or gussian function.
          band,float/int
              exact position of the Fraunhofer line
    return:
          list[sif,reflectance,rmse,B]
              sif, numpy array, sif at the Fraunhofer line for each measurement
              reflectance, numpy array
              rmse, RMSE
              B, numpy array, the parameters of the fitting equation
                  [a,b,c,d,e,f]: a, b, c for polynimal refelectance
                                 d, float, MAX sif value [0,10] mw/m2/nm/sr
                                 e, position of MAX sif value - wl_range, [0,wavelength.size], nm
                                 f, full width at half maximum, [0,wavelength.size*5]
    """
    sif,refl,rmse,B = [],[],[],[]
    data = data.where((data.Wavelength>wl_range[0])&(data.Wavelength<wl_range[1]),drop=True)
    absorb_line_position = np.where(data.Wavelength == data.Wavelength.sel(Wavelength=band,method='nearest').values)
    _nmeas = data.Measures.size
    _wvl = data.Wavelength.values
    _nwvl = _wvl.size
    _x = np.interp(_wvl, (_wvl.min(),_wvl.max()),(-1,1))
    # normalize standard SIF template by the mean value
    _hf = data.hf.values/(np.mean(data.hf.values))
    # create base function by legendre polynomial equations
    _ref_base = np.array([eval_legendre(n, _x) for n in np.arange(6)])
    
    for i in range(0,_nmeas):
        print(data.Measures[i].values)
        sky_spec = data.sky[i].values
        veg_spec = data.veg[i].values
        _X = np.concatenate([_ref_base,(_hf/(veg_spec+0.000001111)).reshape(1,-1)]).T
        _y = np.log(veg_spec+0.000001111) - np.log(sky_spec+0.00000111111)
        _B = np.linalg.lstsq(_X,_y,rcond=-1)
        _sif = _hf * _B[0][-1]
        _refl = (veg_spec - _sif) / (sky_spec+0.0000011111)
        sif.append(_sif[absorb_line_position][0])
        refl.append(_refl[absorb_line_position][0])
        rmse.append(np.sqrt(np.sum(_B[1]**2)/_nwvl))
        B.append(_B[0])
        
    return [sif,refl,rmse,B]

def svd(data,wl_range,band=760,num_vector=20,pow_of_refl = 5):

    """
    Singular Vector Decomposition (Huaize Feng)
    input:
          data,
              xarray dataset
          wl_range,[start,end]
              a window around Fraunhofer lines position, ,about 10 nm
              for fitting a polynomial or gaussian function.
          band,float/int
              exact position of the Fraunhofer line
          num_vector,int
                pca保留的观测维度（个数），相当于去除大气的噪音
          pow_of_refl,int
                反射率所用多项式的最高次数
              Highest power of the polynomial equation denoting reflectance
    return:
          list[sif,reflectance,rmse,B]
              sif, numpy array, sif at the Fraunhofer line for each measurement
              reflectance, numpy array
              rmse, RMSE
              B, numpy array, the parameters of the fitting equation
                  [a,b,c,d,e,f]: a, b, c for polynimal refelectance
                                 d, float, MAX sif value [0,10] mw/m2/nm/sr
                                 e, position of MAX sif value - wl_range, [0,wavelength.size], nm
                                 f, full width at half maximum, [0,wavelength.size*5]
    """
    sif,refl,rmse,B = [],[],[],[]
    data = data.where((data.Wavelength>wl_range[0])&(data.Wavelength<wl_range[1]),drop=True)
    abosorb_line_position = np.where(data.Wavelength == data.Wavelength.sel(Wavelength=band,method='nearest').values)
    _nmeas = data.Measures.size
    _wvl = data.Wavelength.values
    _nwvl = _wvl.size
    _hf = data.hf.values  # 重采样后的标准sif
    
    u, s, vh = np.linalg.svd(data.sky)  # Svd分解----------------------------------------------------------------------------------------------------
    v = -vh

    tmp1=(_wvl-np.mean(_wvl)).reshape(-1,1)
    tmp2=np.arange(pow_of_refl+1)
    p1 = np.power(tmp1, tmp2)*v[0].reshape(-1,1)
    p2 = np.power((_wvl-np.mean(_wvl)).reshape(-1,1), np.arange(pow_of_refl+1))*v[1].reshape(-1,1)
    p3 = v[:,2:num_vector]
    _X = np.concatenate([p1,p2,p3,_hf.reshape(-1,1)],axis=1)
    
    for i in range(0,_nmeas):
        sky_spec = data.sky[i].values
        veg_spec = data.veg[i].values
        _y = veg_spec
        _B = np.linalg.lstsq(_X,_y,rcond=-1)  # https://www.zhihu.com/question/40540185?sort=created
        _sif = _hf * _B[0][-1]
        _refl = (veg_spec - _sif) / (sky_spec + 0.0000001)
        sif.append(_sif[abosorb_line_position][0])
        refl.append(_refl[abosorb_line_position][0])
        rmse.append(np.sqrt(np.sum(_B[1]**2)/_nwvl))
        B.append(_B[0])
    return [sif,refl,rmse,B]


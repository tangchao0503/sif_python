# Multi Processing Version.
# 2022/1/14 14:52

# 唐超：如何将此文件打包成一个exe
#  （1）cd D:\csharp_vs2017\easySif\sifAlgorithm_Python_exe
#  （2）D:\software\Anaconda3\envs\python_36\Scripts\pyi-makespec.exe -F D:\PycharmProjects\sif\sif_retrieval.py
#  （3）在生成的spec文件中加入import sys
#                            sys.setrecursionlimit(5000)
#  （4）D:\software\Anaconda3\envs\python_36\Scripts\pyinstaller.exe  D:\csharp_vs2017\easySif\sifAlgorithm_Python_exe\sif_retrieval.spec

import os
import sys
from glob import glob
import ast
import csv
import sif_methods
import argparse
from multiprocessing import Pool
import multiprocessing
from scipy.interpolate import splev, splrep
import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception as e:
    print(e)
    cmd = 'pip install xarray -i https://pypi.tuna.tsinghua.edu.cn/simple --user'
    os.system(cmd)
    import xarray as xr
finally:
    pass


def intep(standard, x2):
    '''
    插值，根据给定的波长序列将标准SIF重新插值
    输入：标准SIF，两列，一列波长一列值；给定新的波长序列
    输出：插值后的SIF值
    '''
    spl = splrep(standard.index, standard)
    y2 = splev(x2, spl)
    return y2


def sel_method(x):
    '''
    选择方法，
    输入：方法字符串
    输出：方法
    '''
    return {
        'sfld': sif_methods.sfld,
        '3fld': sif_methods.fld3,
        'svd': sif_methods.svd,
        'sfm': sif_methods.sfm,
        'sfm_gaussian': sif_methods.sfm_gaussian,
        'doas': sif_methods.doas
    }.get(x, sif_methods.sfld)


def read_files(f):
    '''
    使用csv.reader读取文件
    '''
    with open(f) as csvfile:
        data = list(csv.reader(csvfile))  # 读csv并转化为list
        data = [[element for element in row if element != ''] for row in data]  # 去掉每行的空字符
        data = np.array(data, dtype=object)  # 转化为np数组
        data = [np.array(d) for d in data]
        csvfile.close()
    return data


def Convert(lst):
    '''
    针对<键名 键值...>组织形式的文件头解析
    '''
    # tc
    #     res_dct = {lst[i]: lst[i + 1] for i in range(0, lst.size-1, 2)}
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst)-1, 2)}
    return res_dct

# parsing


def data_parser(data, dt):
    shp = np.arange(len(data)).tolist()
    _iter = iter(shp)

    # 读取csv的元信息（第一行）
    data[next(_iter)][1:]  # 略过csv的第一行
    header_dict = {'Measure': dt}
    # header_dict = Convert(data[next(_iter)][1:])
    header_dict.update(Convert(data[next(_iter)]))

    # 读取波长等信息
    ls = []
    for i in range(int(header_dict['TotalSpectrometer'])):
        info = Convert(data[next(_iter)][1:])  # 略过第3行
        wvl = np.array(data[next(_iter)][1:])
        info['Wavelength'] = (wvl[wvl != '']).astype('float')
        info.update(header_dict)
#         info = my_dict
        ls.append(info)

    # 读取光谱数据?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    data = data[next(_iter)+1:]
    data = pd.DataFrame(data)
    data[['spec', 'point']] = data[0].str.split('_', expand=True)  #??
    data = data.set_index(['spec', 'point'])
    data = data.iloc[:, 3:]
    data.columns.name = 'Wavelength'
    data = data.astype('float')
    data = data.T.unstack().to_xarray()
    # _ ['spec'] = f
    data['Measures'] = dt
    data = data.expand_dims('Measures')
    return [data, ls]  # data是光谱数据，ls是元信息


def map_files(f):
    '''
    遍历文件
    '''
    dt = pd.to_datetime(f.split('\\')[-1][:-4], format='%Y_%m_%d_%H_%M_%S')
    data = read_files(f)
    data = data_parser(data, dt)
    #     par_ls.append(par)
    return data    # data是列表：第一个是光谱数据，第二个是元信息

# def map_files(i):
#     result = i * i
#     return result


def processing(standard_sif, folder, out_file, pars, data, header, sky_p='P1', method='sfld'):
    '''
    提取算法调用
    '''
    # 参数解析为list
    pars = ast.literal_eval(pars)
    standard_sif = pd.read_csv(standard_sif, index_col=[0])
    point_ls = {}
    ls = []
    # 遍历光谱仪
    for spec in data['spec'].values:
        if "SIF" not in spec:
            continue
        else:
            # 
            _ = data.sel(spec=spec)  # ???????????????
            # 根据光谱仪提取对应的波长
            wvl = header['Wavelength'][spec]
            size = wvl.size
            # 使用波长的长度对数据进行截取（减少多光谱仪数据不一致导致的空值）
            _ = _.isel(Wavelength=xr.DataArray(
                np.arange(0, size), dims="Wavelength"))
            _['Wavelength'] = wvl
        #     _ = _.where((_.Wavelength>731.3)&(_.Wavelength<782),drop=True)

            sky = _.sel(point=sky_p, drop=True).rename('sky')
            for p in _.point:
                if p == sky_p:
                    continue
                else:
                    veg = _.sel(point=p, drop=True).rename('veg')
                    input_each = xr.merge([sky, veg])
                    _hf = intep(standard_sif, input_each.Wavelength.values)  # 将标准sif插值匹配到数据的波长
                    input_each['hf'] = (['Wavelength'], _hf)
                    # 调用方法
                    retr_method = sel_method(method)
                    print('Running {} method on {} of spectrometer {}'.format(
                        method, p.values, spec), flush=True)
                    print('Processing ...', flush=True)
                    sif = retr_method(input_each, *pars)[0]#---------------------------------------------------------------------------------------------
                    point_ls.update({str(p.values): sif})
            _sif = pd.DataFrame(point_ls)
            point_ls = {}
            _sif.index = sky.Measures.values
            _sif.index.name = 'Measures'
            _sif = _sif.to_xarray()
            _sif = _sif.assign_coords(spec=spec)
            ls.append(_sif)
    sif = xr.merge(ls)
    # sif.to_dataframe().to_csv(out_file)

    tmp_tc = sif.to_dataframe()  # tmp_tc是Pandas 数据结构 - DataFrame
    cols = tmp_tc.columns
    rows = tmp_tc.index
    for col in cols:
        if str(tmp_tc[col].dtype) == 'object':  # 如果是字符串就略过
            continue
        for i in range(tmp_tc[col].shape[0]):
            if (tmp_tc[col][i] < 0):
                tmp_tc.loc[rows[i], col] = 0
                # tmp_tc[col][i] = 0  # 会出错：A value is trying to be set on a copy of a slice from a DataFrame

    # for i in range(tmp_tc.shape[0]):  # 行
    #     for j in range(tmp_tc.shape[1]):  # 列
    #         if str(tmp_tc.iloc[i, j].dtype) == 'object':  # 如果是字符串就略过
    #             continue
    #         print(tmp_tc.iloc[i, j])

    tmp_tc.to_csv(out_file)




# # 需要输入的参数
# # 标准SIF曲线
# standard_sif = 'standard_sif.csv'
# # 天空辐射对应的点
# sky_p = 'P1'
# # 存放文件夹
# folder = '2021_12_21'
# # 提取方法需要的输入参数，用括号括起来
# pars = ([740,770],760)

# processing(standard_sif,folder,outfile,pars,sky_p='P1',method='svd')

def parse_args():
    parser = argparse.ArgumentParser(description="SIF retrieval process")
    parser.add_argument("standard_sif")
    parser.add_argument("folder")
    parser.add_argument("outfile")
    parser.add_argument("pars")
    parser.add_argument("sky_p", default='P1')
    parser.add_argument("method", default='sfld')
    args = parser.parse_args()
    return args


def main():
    #  https://blog.csdn.net/maskxxx/article/details/109292698
    #  https://blog.csdn.net/weixin_42146296/article/details/92848315?utm_medium=distribute.pc_relevant.none-task-blog-title-1&spm=1001.2101.3001.4242
    multiprocessing.freeze_support()  # 加入这行代码，用pyinstaller打包后，pool.map才能工作正常

    if (len(sys.argv) > 7):  # 1个本py文件 + 6个参数
        print('sif error: 程序安装路径中不能有空格！')

    # 解析命令行参数
    inputs = parse_args()

    # 读取数据
    cpus = os.cpu_count()
    pool = Pool(cpus)
    # print('tc-----------!', flush=True)
    files = glob(inputs.folder+'/*.csv')
    print('Total files: ', len(files), flush=True)

    print('Reading files ...', flush=True)
    _ = pool.map(map_files, files)
    print('\tdone!', flush=True)

    # 取出光谱数据
    # data = [d[0] for d in _]
    # for x in files:
    #     map_files(files)

    data = [d[0] for d in _]
    data = xr.concat(data, dim='Measures')
    # data.to_netcdf('C:\\fhz\\neibuwenjian\\tangchao\_Data\\temp.nc')

    # 取出元数据
    header = _[0][1]
    _ = []
    header = pd.DataFrame(header).set_index('Model')
    # print(header, flush=True)

    # sif算法处理
    try:
        processing(inputs.standard_sif, inputs.folder, inputs.outfile,
                   inputs.pars, data, header, inputs.sky_p, inputs.method)
    except Exception as e:
        print('sif error: %s' % e, flush=True)
    else:  # 如果打开没有出错
        print('Sif compute Completed!')
    finally:
        pass


if __name__ == '__main__':
    main()

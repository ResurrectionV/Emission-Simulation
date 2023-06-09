import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
from scipy.interpolate import interp1d


temp_df1 = dataDF = pd.read_excel('221201.xlsx',sheet_name="3号")
temp_df2 = dataDF = pd.read_excel('221201.xlsx',sheet_name="5号")
temp_df3 = dataDF = pd.read_excel('221201.xlsx',sheet_name="7号")
temp_df4 = dataDF = pd.read_excel('221201.xlsx',sheet_name="8号")
temp_df5 = dataDF = pd.read_excel('221201.xlsx',sheet_name="9号")
temp_df6 = dataDF = pd.read_excel('221201.xlsx',sheet_name="10号")
# 读取文件
dataDF = pd.concat([temp_df1,temp_df2,temp_df3,temp_df4,temp_df5,temp_df6])
concDF = xr.open_dataset('conc_new.nc') # LSM模拟浓度数据


class EmissionSimulation:
    # 初始化参数 与 设定变量
    def __init__(self, concPath = concDF, obsPath = dataDF):
        self.mVtoPPM = 1/2.5 # mV 到 PPM 的转换
        self.SCFHToPPM = 470/60 # SCFH 到 PPM 的转换

        # 可调整的参数
        self.obsNoise = 0.5 # estimated from blank test, in ppm
        self.obsNoiseRatio = 0.05 # estimated from sensor's datasheet
        self.mdlErrorRatio = 0.2 # estimate (you can vary this, usually 0.2-0.4)
        # 1. 时间间隔选择 可以在20-30分钟， 根据数据时间间隔计算， 假设数据时间间隔为1秒和30分钟间隔， 那么intveral计算为 30*60 = 1800
        # 2. 频率为 frequency秒一次数据， frequency=1，0.5，0。25 为 1秒，2秒，4秒间隔的数据
        # 3. nTime计算在数据里共有多少间隔
        self.interval = 30*60
        self.frequency = 1 
        self.nTime = 10 

        # 可能的排放强度
        self.leakRates = np.arange(1,50) # 排放强度范围设定

        # 目标网格宽度 (1000 * 1000 * 500) = (xyz),步长为5米
        self.ds = 5 

        ## 观测点位信息
        # 数据来源于GPS.txt
        # 0号: 排口，其余为观测点
        # 词典数值为三维坐标（纬度，经度，高度（米））
        # 这个词典是预设值
        self.meta = {'9号': (121.276789, 30.728726, 5.0),
                '10号': (121.276399, 30.728756, 2.5),
                '5号': (121.277073, 30.727993, 4.0),
                '3号': (121.278509, 30.730408, 2.5),
                '8号': (121.278153, 30.730463, 20.0),
                '7号': (121.273118, 30.726355, 15.0),
                '0号': (121.263, 30.708, 20.0)} 
        #以下为文件读取词典，同上数据结构
        self.gpsLoc = {
            '9': (0, 0, 0),
            '10': (0, 0, 0),
            '5': (0, 0, 0),
            '3': (0, 0, 0),
            '8': (0, 0, 0),
            '7': (0, 0, 0),
            '0': (121.276731,30.728845,20.)
        }
    

        ## 观测数据存取
        self.obsData = obsPath

        ## 模拟数据
        self.mdlFile = concPath # lsm模拟数据 nc文档类型 {(x,y,z),浓度}
        self.mdlData = self.mdlFile['conc'].values # 浓度值
        self.mdlX = self.mdlFile['x'].values # 网格里 所有X坐标
        self.mdlY = self.mdlFile['y'].values # 网格里 所有Y坐标
        self.mdlZ = self.mdlFile['z'].values # 网格里 所有Z坐标
        self.dx = self.mdlX[1] - self.mdlX[0] # 网格x轴宽度
        self.dy = self.mdlY[1] - self.mdlY[0] # 网格y轴宽度
        self.dz = self.mdlZ[1] - self.mdlZ[0] # 网格z轴宽度
        self.xBeg = self.mdlX[0] - self.dx/2
        self.yBeg = self.mdlY[0] - self.dy/2
        self.zBeg = self.mdlZ[0] - self.dz/2

        # Remove Background concentration and bias-correction
        self.nSensor = len(self.meta.keys())-1
        self.nleak = len(self.leakRates)
        self.epct = {} # {ID, value of 5% percentile}
        self.sensorIDsO = {} # {ID,Original Tvoc}
        self.sensorIDs = {} # {ID,AboveAmbient Tvoc}
        self.num_dtc = []

        

        ## Perform source characterization model
        # Number of Grid is based on the conc simulation data
        self.nx = 200 
        self.ny = 200 
        #self.QA = np.zeros((2,self.nSensor))              # Quality assurance check output
        self.postPDF = np.ones((self.nx, self.ny, self.nleak)) # posterior pdf
        self.locationPDF = np.zeros((self.nx, self.ny, 2)) # initial location pdf
        self.strengthPDF = np.zeros((self.nleak, 2))  # initial strength pdf
        #self.prior_Uni = np.ones( (self.nx,self.ny,self.nleak) ) / (self.nx*self.ny*self.nleak)  # prior (uniform)

        # 具体读取的是哪一个观测点 根据class后的操作判断
        self.df1 = None # Dataframe of windDirection from sensor #3
        self.df2 = None # Dataframe of windDirection from sensor #5
        self.df3 = None # Dataframe of windDirection from sensor #7
        self.df4 = None # Dataframe of windDirection from sensor #8
        self.df5 = None # Dataframe of windDirection from sensor #9
        self.df6 = None # Dataframe of windDirection from sensor #10

        self.df11 = None # Dataframe of windSpeed from sensor #3
        self.df12 = None # Dataframe of windSpeed from sensor #5
        self.df13 = None # Dataframe of windSpeed from sensor #7
        self.df14 = None # Dataframe of windSpeed from sensor #8
        self.df15 = None # Dataframe of windSpeed from sensor #9
        self.df16 = None # Dataframe of windSpeed from sensor #10

        self.df21 = None # Dataframe of Tvoc from sensor #3
        self.df22 = None # Dataframe of Tvoc from sensor #5
        self.df23 = None # Dataframe of Tvoc from sensor #7
        self.df24 = None # Dataframe of Tvoc from sensor #8
        self.df25 = None # Dataframe of Tvoc from sensor #9
        self.df26 = None # Dataframe of Tvoc from sensor #10

        self.newdf1 = None # 同一时间段不同观测点风向数据合并同一DatafFrame
        self.newdf2 = None # 同一时间段不同观测点风速数据合并同一DatafFrame
        self.TvocDF = None # 同一时间段不同观测点观测浓度数据合并同一DatafFrame

        self.wdMedian = None # 中位数风向
        self.wsMean = None # 平均数风速

        self.gridGPS = {} # 经纬度到网格坐标的词典


        
    # 1.移除背景浓度
    def AboveAmbient(self,lst):
        #print(np.percentile(np.sort(lst),5))
        rt_lst = (lst-np.percentile(np.sort(lst),5))*1000  # Remove background Concentration 减去5% percentile后再进行ppb到ppm的单位转换
        rt_lst = np.where(rt_lst>0,rt_lst,0) #Bias Correction 将负数数据换为0， （ReLU）
        return rt_lst

    # GPS.txt
    def llcoor(self, gpsPath = 'GPS.txt'):
        with open(gpsPath, "r") as file:
            lines = file.readlines()

        loc_name = []

        for line in lines:
            split_line = line.split(',')
            if len(split_line) >= 5:
                value_2nd = float(split_line[2].strip()) # Latitude
                value_3rd = float(split_line[3].strip()) # Longitude
                value_4th = float(split_line[4].strip()) # Height
                loc_name.append(split_line[1].strip()) # Sensor location name
                key = split_line[0].strip()
                self.gpsLoc[key] = (value_2nd, value_3rd, value_4th)

    # 将词典key转换成list数据格式
    def getKeysinList(self, target):
        targetList = []
        for key in target.keys():
            targetList.append(key)
        return targetList

    # 1.计算在lsm坐标系下的观测点与排口坐标 2. pout=True则作图
    # Currently, we use (x,y) \in [0,1000]x[-500,500]
    # Grid Spacing = 5 meters
    def locationPlotO(self,pout = False):
        xyz_coordinates = []
        for key, (latitude, longitude, altitude) in self.gpsLoc.items():
            xyz_coordinates.append((longitude, latitude, altitude))
       

        x_coordinates, y_coordinates, z_coordinates = zip(*xyz_coordinates)
        # Calculate distances in meters
        distances = []
        for i in range(len(xyz_coordinates)):
            for j in range(i+1, len(xyz_coordinates)):
                # Compute distance with x and y only
                distance = geodesic(xyz_coordinates[i][:2], xyz_coordinates[j][:2]).meters
                distances.append(distance)

        # Scale x, y, and z coordinates to fit the desired range
        x_scaled = [(x - min(x_coordinates)) / (max(x_coordinates) - min(x_coordinates)) * 1000 for x in x_coordinates]
        y_scaled = [(y - min(y_coordinates)) / (max(y_coordinates) - min(y_coordinates)) * 1000 - 500 for y in y_coordinates]

        # 已是lsm坐标则无需转换
        #z_scaled = [(z - min(z_coordinates)) / (max(z_coordinates) - min(z_coordinates)) * 500 for z in z_coordinates]

        keylst = list(self.meta.keys())
        # 存入新的词典
        for i in range(7):
            self.gridGPS[keylst[i]] = (x_scaled[i],y_scaled[i],z_coordinates[i])

        if pout:
            # Plot the locations in the xy plane
            plt.scatter(x_scaled, y_scaled)

            # Add labels for each point
            gpsList_key = self.getKeysinList(self.gpsLoc)
            for i, (x, y) in enumerate(zip(x_scaled, y_scaled)):
                plt.text(x, y,gpsList_key[i])

            # Set axis labels and title
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.title('Locations')

            # Set the range for the x and y axes
            plt.xlim(0, 1000)
            plt.ylim(-500, 500)

            # Display the plot
            plt.show()


    def extractFeature(self,name,colIndex,featureName):
        # 1. 对数据进行拉格朗日插值，原先4秒一个数据点，现1秒一个数据点
        # 2. 针对观测点name计算和返还dataframe
        time_str = list(self.obsData.iloc[np.array(self.obsData.iloc[:,0] == name),2])
        start_time_str = time_str[0]
        end_time_str = time_str[-1]
        start_time_obj = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_time_obj = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
        total_seconds = (end_time_obj - start_time_obj).total_seconds()


        wind_direction = list(self.obsData.iloc[np.array(self.obsData.iloc[:,0] == name),colIndex])

        # Convert time strings to datetime objects
        time_obj = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in time_str]

        # Convert datetime objects to numerical timestamps
        time_num = np.array([(t - datetime(1970, 1, 1)).total_seconds() for t in time_obj])

        # Create an interpolation function
        interp_func = interp1d(time_num, wind_direction, kind='linear', fill_value='extrapolate')

        # Generate timestamps for interpolation
        interp_time = np.linspace(time_num[0], time_num[-1], num=int(total_seconds) + 1)  # Interpolation timestamps

        # Interpolate wind directions
        interpolated_wind_direction = interp_func(interp_time)

        # Convert interpolation timestamps back to datetime objects
        interp_time_obj = [datetime.utcfromtimestamp(t) for t in interp_time]

        return pd.DataFrame({'Time':interp_time_obj,featureName:interpolated_wind_direction})
    

    # 获得新数据帧，使得所有数据所处同一时间段且间隔相同
    def overlappingTimestamp(self,dF1,dF2,dF3,dF4,dF5,dF6,featureName):
        timestamp_column = 'Time'

        # 获得起始时间与结束时间
        start_time = max(df[timestamp_column].iloc[0] for df in [dF1,dF2,dF3,dF4,dF5,dF6])
        end_time = min(df[timestamp_column].iloc[-1] for df in [dF1,dF2,dF3,dF4,dF5,dF6])

        # 存放数据
        rowNum1 = []
        rowNum2 = []
        rowNum3 = []
        rowNum4 = []
        rowNum5 = []
        rowNum6 = []

        # Loop through the DataFrames and find the row numbers for the overlapping timestamps for wind direction
        for i, df in enumerate([dF1,dF2,dF3,dF4,dF5,dF6]):
            common_rows = df.index[(df[timestamp_column] >= start_time) & (df[timestamp_column] <= end_time)]
            
            # Store row numbers in the corresponding list
            if i == 0:
                rowNum1 = common_rows.tolist()
            elif i == 1:
                rowNum2 = common_rows.tolist()
            elif i == 2:
                rowNum3 = common_rows.tolist()
            elif i == 3:
                rowNum4 = common_rows.tolist()
            elif i == 4:
                rowNum5 = common_rows.tolist()
            elif i == 5:
                rowNum6 = common_rows.tolist()

        newDF = pd.DataFrame({'Time':dF1.iloc[rowNum1]['Time'],
                            featureName+'1':np.array(dF1.iloc[rowNum1][featureName]).tolist(),
                            featureName+'2':np.array(dF2.iloc[rowNum2][featureName]).tolist(),
                            featureName+'3':np.array(dF3.iloc[rowNum3][featureName]).tolist(),
                            featureName+'4':np.array(dF4.iloc[rowNum4][featureName]).tolist(),
                            featureName+'5':np.array(dF5.iloc[rowNum5][featureName]).tolist(),
                            featureName+'6':np.array(dF6.iloc[rowNum6][featureName]).tolist()})
        return newDF
    



    def get_wind_rose_obs_conc(self,wd,conc,nvalid=30,Thre=2.5):
        # 对每一个角度进行浓度检测
        # 角度有部分重叠 （0-10，5-15...）
        thetaDelta = 5 # degree
        thetas = np.arange(5,360,5)
        # concMean： 角度范围内平均浓度
        # concStd： 角度范围内浓度标准差 mu +- 2*std
        # concDir： 角度数据列 in Degree
        concMean, concStd, concDir = [], [], []

        for theta in thetas:
            indexs = np.logical_and(wd > theta-thetaDelta, wd < theta+thetaDelta)
            if sum(indexs) > nvalid:
                iConc = np.mean(conc[indexs])
                #print(theta, iConc)
                if iConc > Thre:
                    concDir.append( theta )
                    concStd.append( np.std( conc[indexs] ) )
                    concMean.append( iConc )
                    #print(f'concDir:{concDir},concStd:{concStd},concMean:{concMean}')
                    #print()
        return concMean, concStd, concDir


    # 坐标转换以及图像绘制
    '''''
    def new_coordinates_and_plot(self,x0, y0, x1, y1, theta, plot_show = False):
        # Convert theta from degrees to radians
        theta_rad = np.radians(270 - theta)
        # Translate target such that source is at origin
        x1_trans, y1_trans = x1 - x0, y1 - y0

        # Rotate coordinates (translation to new frame)
        x_new = x1_trans * np.cos(theta_rad) + y1_trans * np.sin(theta_rad)
        y_new = -x1_trans * np.sin(theta_rad) + y1_trans * np.cos(theta_rad)

        if plot_show:

            # Calculate wind direction vectors in original frame
            start_x = x1 - 300 * np.cos(theta_rad)
            start_y = y1 - 300 * np.sin(theta_rad)

            wddx = 300 * np.cos(theta_rad)
            wddy = 300 * np.sin(theta_rad)

            

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # Original frame (plot points and wind direction)
            ax[0].plot(x0, y0, 'ko') # Source in black
            ax[0].plot(x1, y1, 'ro') # Target in red
            ax[0].quiver(start_x,start_y,wddx,wddy, angles='xy', scale_units='xy', scale=1, color='b')
            ax[0].set_xlim(0, 1000)
            ax[0].set_ylim(-500, 500)
            ax[0].grid()
            ax[0].set_title('Original Frame')
            ax[0].annotate(f'({x0}, {y0})', (x0, y0), textcoords="offset points", xytext=(-10,10), ha='center')
            ax[0].annotate(f'({x1}, {y1})', (x1, y1), textcoords="offset points", xytext=(-10,10), ha='center')

            # New frame (cover the new coordinate of x1, y1)
            ax[1].quiver(0, 0, 500, 0, angles='xy', scale_units='xy', scale=1, color='b') # wind direction in new frame
            ax[1].plot(0, 0, 'ko') # new source in black
            ax[1].plot(x_new, y_new, 'ro') # new target in red

            x_max = max(1000, x_new + 100) 
            y_max = max(500, y_new + 100)
            y_min = min(-500, y_new - 100)

            ax[1].set_xlim(-x_max, x_max)
            ax[1].set_ylim(y_min, y_max)
            ax[1].grid()
            ax[1].set_title('New Frame')
            ax[1].annotate('(0, 0)', (0, 0), textcoords="offset points", xytext=(-10,10), ha='center')
            ax[1].annotate(f'({x_new:.2f}, {y_new:.2f})', (x_new, y_new), textcoords="offset points", xytext=(-10,10), ha='center')

            # Show the plots
            plt.show()

        return x_new, y_new
        '''''

    # 风处理
    def WindProcess2d(self,wsArr,wdArr):
        # 输入：
        # 风速
        # 风向
        # 输出：
        # ustar: friction velocity, in m/s
        # su: normalized std of u velocity by ustar, non-dimensionalized
        # sv: normalized std of v velocity by ustar, non-dimensionalized
        # mdirDeg: main wind dir during this time interval, in deg
        Ux = wsArr * np.cos(np.radians(wdArr))
        Uy = wsArr * np.sin(np.radians(wdArr))

        
        # Rotation
        mws2x = np.mean(Ux)  # mean u
        mws2y = np.mean(Uy)  # mean v

        mdir = np.arctan2(mws2y, mws2x)
        mdir = (mdir + 2 * np.pi) % (2 * np.pi)  # Convert to range [0, 2pi]
        mdirDeg = np.degrees(mdir)  # Convert to degrees
        mdirDeg = (360 - mdirDeg + 90) % 360  # Convert to meteorological convention (degrees from North)\

        CRA = np.cos(mdir)
        SRA = np.sin(mdir)

        nws2x = Ux*CRA + Uy*SRA
        nws2y = -Ux*SRA + Uy*CRA

        ws2x = nws2x  # velocity component in mean wind direction (u)
        ws2y = nws2y  # horizontal velocity component perpendicular to mean wind (v)

        # calculate ustar
        ustar = np.sqrt(np.abs(np.mean((ws2x - np.mean(ws2x)) * (ws2y - np.mean(ws2y)))))  # friction velocity
        su = np.std(ws2x) / ustar  # std of u
        sv = np.std(ws2y) / ustar  # std of v

        return ustar, ws2x, ws2y, su, sv, mdirDeg
    
    # 坐标变换
    def transformation(self,x0,y0,x1,y1,dirDeg):
        # (x0,y0) 排口位置
        # (x1,y1) 选取的点位置
        # dirDeg 观测点检测到的风向
        # 90 deg is account for wind blowing towards north (y axis) at 0 deg.
        # 180 deg is used for Azi_3D data (0 deg for the WD_2d), these two sensors have different definition for WD (blow towards or away). 
        WD_rad = np.deg2rad(270 - dirDeg)

        W = np.array([[np.cos(WD_rad), np.sin(WD_rad)],
                      [-np.sin(WD_rad), np.cos(WD_rad)]])
        v = np.array([x1-x0,y1-y0])
        v_w = np.matmul(W, v)
        return v_w[0],v_w[1]
    

    # Perform Source Character Model
    # ustar_est: False -- friction velocity 则使用10%的平均风速估算
    # ustar_est: True -- friction velocity 则在风处理函数里计算
    def scModel(self, ustar_est = False):
        # self.QA -- Quality Assurance check Output
        # self.postPDF -- Initialized posterior PDF
        # self.locationPDF -- Initial location PDF
        # self.strengthPDF -- Initial Strength PDF
        # self.prior_Uni -- Prior Uniform PDF
        temp_loc = list(self.gridGPS.values())[-1]
        #nTm = []
        lRate = []

        for tm in range(190): # self.nTime -- number of 30 mins intervals from the data  #self.nTime
            beg = int(tm*self.frequency*self.interval) # begin row number
            end = int(beg + self.frequency*self.interval) # end row number
            # Processing Wind Measurement
            # Input: 
            # self.wdMedian: wind direction in degree
            # self.wsMean: wind speed, in m/s

            # Output:
            # ustar: friction velocity, in m/s
            # u: mean u
            # v: mean v
            # su: normalized std of u velocity by ustar, non-dimensionalized
            # sv: normalized std of v velocity by ustar, non-dimensionalized
            # mdirDeg: main wind dir during this time interval, in deg
            ustar,u,v,su,sv,mdirDeg = self.WindProcess2d(self.wsMean[beg:end],self.wdMedian[beg:end])
            if not ustar_est: # ustar_est = False, we should estimate ustar by 10% of windspeed
                ustar = np.mean( self.wsMean[beg:end] )/10. # ustar is estimated by 10% of windspeed
            # Assuming neutral atm condition.
            #L = 9999
            print(f'Epoch {tm+1}:')
            print()
            #print(f'ustar: {ustar}, sd_u: {su}, sd_v: {sv}, mainDirection: {mdirDeg}')
            for i in range(6):
                sensorName = list(self.gridGPS.keys())[i]
                sensorInfo = list(self.gridGPS.values())[i] # snesorInfo is a tuple which contains (x,y,z) of sensor i
                #print(f'观测点: {sensorName}, 观测位置: {sensorInfo}')
                
                # Perform conditional averaging
                # Input:
                # self.wdMedian[beg:end] # wind direction from certain time interval
                # self.TvocDF[beg:end,i] # concentration of ith sensor from certain time interval
                # Output
                # Output:
                # concObs: Bin averaged concentration 
                # concStd: uncertainty related with the bin-averaging
                # concDir: number of data points fall into a bin
                concObs, concStd, concDir = self.get_wind_rose_obs_conc(self.wdMedian[beg:end], self.TvocDF[beg:end,i])
                if len( concObs ) > 0:  

                    for xstep in range(self.nx):
                        for ystep in range(self.ny): # Traverse the conc positions
                            concMdl = np.zeros( (len(concObs)) )
                            SouLoc_x = (xstep + 1) * self.ds
                            SouLoc_y = -500 + (ystep + 1) * self.ds

                            for dirInd,dirEle in enumerate( concDir ): # Each time extract the index of the element and the direction value
                                ssdx,ssdy = self.transformation(temp_loc[0],temp_loc[1],SouLoc_x,SouLoc_y,dirEle) # Source to sensor distance for x and y
                                #print(f'({ssdx},{ssdy})')
                                if ssdx > 0:
                                    ssdy = abs(ssdy)
                                    xIndex = int( (ssdx + 0.01 - self.xBeg)/self.dx -1 ) 
                                    yIndex = int( (ssdy + 0.01 )/self.dy -1 ) # absolute value of ssdy?
                                    zIndex = int( (sensorInfo[2] + 0.01 - self.zBeg)/self.dz -1) # simulation hgt = sensor height?
                                    concMdl[dirInd] = self.mdlData[zIndex, yIndex, xIndex]*0.5/ustar*self.SCFHToPPM
                                    
                            diff = np.zeros((self.nleak))
                            for jj in range(self.nleak):
                                # calculate the difference between expected and measured concentration
                                diff[jj] = np.mean(abs(concObs - concMdl*self.leakRates[jj])) # 获取平均差值

                            sigmaE = max(concStd) + max(self.obsNoise, self.obsNoiseRatio*np.mean(concObs)) + self.mdlErrorRatio*np.mean(concMdl)*self.leakRates
                            likeP = np.exp(-0.5*(diff/sigmaE)**2)/sigmaE/np.sqrt(2*np.pi) # likelihood 用的高斯分布
                            self.postPDF[xstep, ystep, :] = self.postPDF[xstep, ystep, :]*likeP
                #break
            
            dim_indx = tm%2
            # 标准化posterior PDF
            self.postPDF = self.postPDF/np.sum(self.postPDF)

            # marginalize for source location (x,y)
            self.locationPDF[:, :, dim_indx] = np.sum(self.postPDF, axis=2)
            self.locationPDF[:, :, dim_indx] = self.locationPDF[:, :, dim_indx]/np.sum( self.locationPDF[:, :, dim_indx]) # sum = 1.

            # marginalize for source location (x,y)
            self.strengthPDF[:, dim_indx] = np.sum(self.postPDF, axis=(0, 1))
            self.strengthPDF[:, dim_indx] = self.strengthPDF[:, dim_indx]/np.sum( self.strengthPDF[:, dim_indx]) # sum = 1.
        
            if (tm+1)%2 == 0:
                bestLeak = np.sum(self.strengthPDF[:, 1]*self.leakRates)
                bestLeakStd = np.sqrt(np.sum(self.strengthPDF[:, -1]*(self.leakRates-bestLeak)**2))
                self.postPDF = np.ones((self.nx, self.ny, self.nleak)) # posterior pdf
                self.locationPDF = np.zeros((self.nx, self.ny, 2)) # initial location pdf
                self.strengthPDF = np.zeros((self.nleak, 2))  # initial strength pdf
                print('Estimated Leak Rate: ', bestLeak)
                print('Leak rate Uncertainty: ',  bestLeakStd)   
                lRate.append(bestLeak)
        fig = plt.figure()
        plt.plot(lRate)
        fig.suptitle('Emission Rate Trajectory', fontsize=20)
        plt.xlabel('Steps(1 hours)', fontsize=18)
        plt.ylabel('Emission Rate', fontsize=16)
        plt.show()
        
        
                   
                

sim1 = EmissionSimulation()
sim1.llcoor()
sim1.df1 = sim1.extractFeature('3号',7,'windDirection')
sim1.df2 = sim1.extractFeature('5号',7,'windDirection')
sim1.df3 = sim1.extractFeature('7号',7,'windDirection')
sim1.df4 = sim1.extractFeature('8号',7,'windDirection')
sim1.df5 = sim1.extractFeature('9号',7,'windDirection')
sim1.df6 = sim1.extractFeature('10号',7,'windDirection')

sim1.df11 = sim1.extractFeature('3号',6,'windSpeed')
sim1.df12 = sim1.extractFeature('5号',6,'windSpeed')
sim1.df13 = sim1.extractFeature('7号',6,'windSpeed')
sim1.df14 = sim1.extractFeature('8号',6,'windSpeed')
sim1.df15 = sim1.extractFeature('9号',6,'windSpeed')
sim1.df16 = sim1.extractFeature('10号',6,'windSpeed')

sim1.df21 = sim1.extractFeature('3号',8,'Tvoc')
sim1.df22 = sim1.extractFeature('5号',8,'Tvoc')
sim1.df23 = sim1.extractFeature('7号',8,'Tvoc')
sim1.df24 = sim1.extractFeature('8号',8,'Tvoc')
sim1.df25 = sim1.extractFeature('9号',8,'Tvoc')
sim1.df26 = sim1.extractFeature('10号',8,'Tvoc')

sim1.newdf1 = sim1.overlappingTimestamp(sim1.df1,sim1.df2,sim1.df3,sim1.df4,sim1.df5,sim1.df6,'windDirection')
sim1.newdf2 = sim1.overlappingTimestamp(sim1.df11,sim1.df12,sim1.df13,sim1.df14,sim1.df15,sim1.df16,'windSpeed')
sim1.TvocDF = sim1.overlappingTimestamp(sim1.df21,sim1.df22,sim1.df23,sim1.df24,sim1.df25,sim1.df26,'Tvoc')
sim1.newdf1



sim1.newdf1 = sim1.newdf1[0:342003-3]
sim1.newdf2 = sim1.newdf2[0:342003-3]
sim1.TvocDF = sim1.TvocDF[0:342003-3]

sim1.TvocDF = np.array(sim1.TvocDF.iloc[:,1::])
sim1.wdMedian = np.median(np.array(sim1.newdf1.iloc[:,1::]),axis=1)
sim1.wsMean = np.array(sim1.newdf2.iloc[:,1::]).mean(axis=1)

# location plot
# sim1.locationPlotO() 

for i in range(6):
    sim1.TvocDF[:,i] = sim1.AboveAmbient(sim1.TvocDF[:,i])

sim1.locationPlotO(pout=False)



sim1.newdf1 = sim1.newdf1[0:342003-3]
sim1.newdf2 = sim1.newdf2[0:342003-3]
sim1.TvocDF = sim1.TvocDF[0:342003-3]

sim1.TvocDF = np.array(sim1.TvocDF.iloc[:,1::])
sim1.wdMedian = np.median(np.array(sim1.newdf1.iloc[:,1::]),axis=1)
sim1.wsMean = np.array(sim1.newdf2.iloc[:,1::]).mean(axis=1)

# location plot
# sim1.locationPlotO() 

for i in range(6):
    sim1.TvocDF[:,i] = sim1.AboveAmbient(sim1.TvocDF[:,i])

sim1.locationPlotO(pout=False)


sim1.scModel()
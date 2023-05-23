# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:59:09 2023

@author: Sherry
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import datetime
from tqdm import tqdm



##计算指标
def calc_vUvD(mkt_data, nL=15):    
    high = mkt_data['high']
    low = mkt_data['low']
    """ 计算指标 """
    # vf：分型顶、分型底标志
    vf = (   (high.shift(1)>high.shift(2)) & (high.shift(1)>high) \
           & (low.shift(1)>low.shift(2)) & (low.shift(1)>low) \
         ).astype(int) - \
         (   (high.shift(1)<high.shift(2)) & (high.shift(1)<high) \
           & (low.shift(1)<low.shift(2)) & (low.shift(1)<low) \
         ).astype(int)
    vf = vf.shift(-1)
    vf[vf==0] = None        
    # vH、vL：分型高低点连续变量
    vH = (high[vf==1] * (vf==1) ).shift(2).fillna(method='ffill')
    vL = (low[vf==-1] * (vf==-1)).shift(2).fillna(method='ffill')
    # vU、vD：分型通道
    vU = vH.rolling(nL).max()
    vD = vL.rolling(nL).min()
    """ 赋值 """
    mkt_data['vf'] = vf
    mkt_data['vH'] = vH
    mkt_data['vL'] = vL
    mkt_data['vU'] = vU
    mkt_data['vD'] = vD
    return mkt_data
m15_300 = pd.read_hdf('IF8888.CCFX.h5')#数据来源：聚宽
m15_300 = calc_vUvD(m15_300, 15)




##计算信号
def calc_signal(mkt_data):
    close = mkt_data['close']
    vU = mkt_data['vU']
    vD = mkt_data['vD']
    #计算信号 
    signals = (close>=vU).astype(int) - (close<=vD).astype(int)
    signals[signals==0]=None
    #赋值 
    mkt_data['signal'] = signals
    return mkt_data

##计算止损价
def calc_sl_price(mkt_data, default_af=0.02 , stepsize_af=0.02, max_af=0.2):
    sl_prices = []
    pre_sig = None
    open_sig = None
    for sig, h, l in zip(mkt_data['signal'], mkt_data['high'], mkt_data['low']):
        # 当期产生了信号
        if sig==1:  # 多头信号
            sl_price = l
            maxHminL = h
            curr_af = default_af
            open_sig = 1
        elif sig==-1: # 空头信号
            sl_price = h
            maxHminL = l
            curr_af = default_af
            open_sig = -1
        else: # sig==None，当期未产生信号
            if open_sig ==1: # 已多头入场
                if h>maxHminL: # 创新高
                    maxHminL=h
                    curr_af +=stepsize_af
                    curr_af = curr_af if curr_af<=max_af else max_af
                sl_price = pre_sl_price + (maxHminL-pre_sl_price) * curr_af
            elif open_sig ==-1:# 已空头入场
                if l<maxHminL: # 创新低
                    maxHminL=l
                    curr_af +=stepsize_af
                    curr_af = curr_af if curr_af<=max_af else max_af
                sl_price = pre_sl_price - (pre_sl_price-maxHminL) * curr_af
            else: # 首单还未开时
                sl_price = None
        pre_sl_price = sl_price
        sl_prices.append(sl_price)
    mkt_data['sl_price'] = sl_prices
    mkt_data['sl_price'] = mkt_data['sl_price'].shift(1)
    return mkt_data



##计算持仓
def calc_position(mkt_data, shift_period=1, otime_limit=None, etime_limit=None, 
                  is_consider_open=True, is_consider_sl=False, is_daily_close=False,
                  comm = None, leverage=None):
   
    signal = mkt_data['signal']
    sl_price = mkt_data['sl_price']
    position = signal.fillna(method='ffill').shift(shift_period).fillna(0)

    """ A. 是否考虑止损，会根据止损结果重新过滤一遍signal，进而确定position列 """
    if is_consider_sl:        
        """ 1. 计算is_sl：
                若持有空仓，sl-h<=0则止损了
                若持有多仓，l-sl<=0则止损了
        """
        is_sl = (  (-sl_price * np.sign(position))
                 - (mkt_data['high']*(position<0)) 
                 + (mkt_data['low']*(position>0)))<=0
        """ 2. 考虑止损结果生成position """
        #  将止损当期signal设为0（当期signal已为1 or -1，则不变）
        signal_consider_sl = signal.copy()
        signal_consider_sl[(signal_consider_sl.isnull()) & (is_sl)] = 0 
        position = signal_consider_sl.fillna(method='ffill').shift(shift_period).fillna(0)
        """ 3. 空仓期间的is_sl设为False """
        is_sl[position==0] = False
    else:
        is_sl = False
        
    """ B. 是否每天otime前空仓 """
    if otime_limit:
        position = position[mkt_data['time']<=otime_limit]=0
    
    """ C. 是否每天etime后空仓 """
    if etime_limit:
        position = position[mkt_data['time']>etime_limit]=0
    
    """ 生成hold_in_price, hold_out_price """
    # 1. 先初始化 period_out_price = close； period_in_price=close.shift(1)
    open_p, close = mkt_data['open'], mkt_data['close']
    hold_in_price =  close.shift(1).copy()
    hold_in_price[close.index[0]] = open_p.values[0]
    hold_out_price = close.copy()

    # 2. 若考虑持仓信号到第二天开盘才能操作
    if is_consider_open:
        # 2-1. 仓位变动第一期的period_in_price为当日open
        hold_in_price[abs(position - position.shift(1).fillna(0))>0] = open_p
        # 2-2. 仓位变动前最后一期的period_out_price为第二日open (除了整个序列最后一天不变)
        hold_out_price[abs(position - position.shift(-1).fillna(0))>0] = open_p.shift(-1)
        hold_out_price[hold_out_price.index[-1]] = close.values[-1]
    
    # 3. 若考虑止损 （mkt_data要先搭配calc_position()函数，计算得到is_sl列）
    if is_consider_sl:
        # 若当期止损了，hold_out_price设为止损价
        #（若多头，止损为open、sl_price里较低的； 若空头，止损为open、sl_price里较高的）
        hold_out_price[is_sl] = (position>0) * np.minimum(open_p, sl_price) + (position<0) * np.maximum(open_p, sl_price)
        # 若上期止损了，当期hold_in_price设为开盘价
        sl_next_idxes = mkt_data[is_sl].index.values+1
        sl_next_idxes = sl_next_idxes[sl_next_idxes<=mkt_data.index.values[-1]]
        hold_in_price.loc[sl_next_idxes] = open_p
    
    # 4. 是否每日收盘前平仓
    if is_daily_close:
        daily_last_idxes = mkt_data.drop_duplicates(subset='date', keep='last').index
        hold_out_price.loc[daily_last_idxes.values] = close
        daily_first_idxes = mkt_data.drop_duplicates(subset='date', keep='first').index
        hold_out_price.loc[daily_first_idxes.values] = open_p
        
    """ 考虑换仓手续费 """ 
    if comm:
        # 换仓后首期 
        if is_consider_sl: # （有止损：仓位相比上期有变动 或 上期止损，当期仓位不为0）
            is_open_pos = ( (abs(position-position.shift(1))>0)|(is_sl.shift(1)) ) * (abs(position)>0)
        else:              # （无止损：仓位相比上期有变动，            当期仓位不为0）
            is_open_pos = (abs(position-position.shift(1))>0) * (abs(position)>0)
        open_pos_comm_perc = 1/(1+comm * is_open_pos)
        # 换仓前末期
        if is_consider_sl: # （有止损：下期仓位有变动 或 当期止损，当期仓位不为0）
            is_close_pos = ( (abs(position-position.shift(-1))>0)|(is_sl) ) * (abs(position)>0)
        else:             # （无止损：下期仓位有变动             ，当期仓位不为0）
            is_close_pos = (abs(position-position.shift(-1))>0) * (abs(position)>0)
        close_pos_comm_perc = (1-comm * is_close_pos)  
    else:
        open_pos_comm_perc = 1.0
        close_pos_comm_perc = 1.0
    
    """ 考虑杠杆 """
    if leverage:
        position *= leverage
    
    """ 赋值 """
    mkt_data['position'] = position
    mkt_data['is_sl'] = is_sl
    mkt_data['hold_in_price'] = hold_in_price * abs(np.sign(position))
    mkt_data['hold_out_price'] = hold_out_price * abs(np.sign(position))
    mkt_data['open_pos_comm_perc'] = open_pos_comm_perc
    mkt_data['close_pos_comm_perc'] = close_pos_comm_perc
    return mkt_data



##回测评估
def statistic_performance(mkt_data, 
                          r0=0.03, 
                          data_period=1440,
                          is_consider_sl=False):

    position = mkt_data['position']
    hold_in_price = mkt_data['hold_in_price']
    hold_out_price = mkt_data['hold_out_price']
    is_sl = mkt_data['is_sl']
    open_pos_comm_perc = mkt_data['open_pos_comm_perc']
    close_pos_comm_perc = mkt_data['close_pos_comm_perc']
    
    d_first = mkt_data['date'].values[0]
    d_last = mkt_data['date'].values[-1]
    d_period = datetime.datetime.strptime(d_last, '%Y-%m-%d') - datetime.datetime.strptime(d_first, '%Y-%m-%d')
    y_period = d_period.days / 365

    hold_r = position * (hold_out_price/hold_in_price-1)
    # 考虑换仓成本 
    hold_r = open_pos_comm_perc * close_pos_comm_perc * (1+hold_r) - 1
    hold_r.fillna(0.0, inplace=True)
    hold_win = hold_r>0
    hold_cumu_r = (1+hold_r).cumprod() - 1
    drawdown = (hold_cumu_r.cummax()-hold_cumu_r)/(1+hold_cumu_r).cummax()    
    ex_hold_r= hold_r-r0/(250*1440/data_period)
    
    mkt_data['hold_r'] = hold_r
    mkt_data['hold_win'] = hold_win
    mkt_data['hold_cumu_r'] = hold_cumu_r
    mkt_data['drawdown'] = drawdown
    mkt_data['ex_hold_r'] = ex_hold_r
    
    v_hold_cumu_r = hold_cumu_r.values[-1]

    v_pos_hold_times= 0 
    v_pos_hold_win_times = 0
    v_pos_hold_period = 0
    v_pos_hold_win_period = 0
    v_neg_hold_times= 0 
    v_neg_hold_win_times = 0
    v_neg_hold_period = 0
    v_neg_hold_win_period = 0
    v_pos_sl_times = 0 
    v_neg_sl_times = 0

    for w, r, pre_pos, pos, is_lastp_sl in zip(hold_win, hold_r, position.shift(1), position, is_sl.shift(1)):
        if pre_pos!=pos or is_lastp_sl: # 当周期有换仓 or 上期止损（先结算上次持仓，再初始化本次持仓）
            if pre_pos == pre_pos: # pre_pos非空（为空则是循环第一次，无需结算）
                # 结算上一次持仓
                if pre_pos>0: # 多仓
                    v_pos_hold_times += 1
                    v_pos_hold_period += tmp_hold_period
                    v_pos_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r>0:
                        v_pos_hold_win_times+=1
                    if is_lastp_sl:
                        v_pos_sl_times += 1
                elif pre_pos<0: # 空仓
                    v_neg_hold_times += 1      
                    v_neg_hold_period += tmp_hold_period
                    v_neg_hold_win_period += tmp_hold_win_period
                    if tmp_hold_r>0:                    
                        v_neg_hold_win_times+=1
                    if is_lastp_sl:
                        v_neg_sl_times += 1
            # 初始化持仓（每次关仓结算后，或循环第一次时）
            tmp_hold_r = 0
            tmp_hold_period = 0 
            tmp_hold_win_period = 0
        if abs(pos)>0:
            tmp_hold_period += 1
            if r>0:
                tmp_hold_win_period += 1
            if abs(r)>0:
                tmp_hold_r = (1+tmp_hold_r)*(1+r)-1       

    v_hold_period = (abs(position)>0).sum()
    v_hold_win_period = (hold_r>0).sum()
    v_max_dd = drawdown.max()
    
    v_annual_ret = (1+v_hold_cumu_r) ** (1/y_period) - 1
    v_annual_std = ex_hold_r.std() * np.sqrt( len(mkt_data)/y_period ) 
    v_sharpe= v_annual_ret / v_annual_std

    """ 生成Performance DataFrame """
    if is_consider_sl:
        performance_cols = ['累计收益', 
                            '多仓次数', '多仓成功次数', '多仓胜率', '多仓平均持有期',
                                        '多仓止损次数', '多仓止损率',
                            '空仓次数', '空仓成功次数',  '空仓胜率', '空仓平均持有期', 
                                        '空仓止损次数', '空仓止损率',
                            '周期胜率', '最大回撤', '年化收益/最大回撤',
                            '年化收益', '年化标准差', '年化夏普'
                           ]
        performance_values = ['{:.2%}'.format(v_hold_cumu_r),
                              v_pos_hold_times, v_pos_hold_win_times,
                                                '{:.2%}'.format(v_pos_hold_win_times/v_pos_hold_times), 
                                                '{:.2f}'.format(v_pos_hold_period/v_pos_hold_times),
                              v_pos_sl_times,   '{:.2%}'.format(v_pos_sl_times/v_pos_hold_times),
                              v_neg_hold_times, v_neg_hold_win_times,
                                                '{:.2%}'.format(v_neg_hold_win_times/v_neg_hold_times), 
                                                '{:.2f}'.format(v_neg_hold_period/v_neg_hold_times),
                              v_neg_sl_times,   '{:.2%}'.format(v_neg_sl_times/v_neg_hold_times),
                              '{:.2%}'.format(v_hold_win_period/v_hold_period), 
                              '{:.2%}'.format(v_max_dd), 
                              '{:.2f}'.format(v_annual_ret/v_max_dd),
                              '{:.2%}'.format(v_annual_ret), 
                              '{:.2%}'.format(v_annual_std), 
                              '{:.2f}'.format(v_sharpe)
                             ]
    else:
        performance_cols = ['累计收益', 
                            '多仓次数', '多仓成功次数', '多仓胜率', '多仓平均持有期', 
                            '空仓次数', '空仓成功次数',  '空仓胜率', '空仓平均持有期', 
                            '周期胜率', '最大回撤', '年化收益/最大回撤',
                            '年化收益', '年化标准差', '年化夏普'
                           ]
        performance_values = ['{:.2%}'.format(v_hold_cumu_r),
                              v_pos_hold_times, v_pos_hold_win_times,
                                                '{:.2%}'.format(v_pos_hold_win_times/v_pos_hold_times), 
                                                '{:.2f}'.format(v_pos_hold_period/v_pos_hold_times),
                              v_neg_hold_times, v_neg_hold_win_times,
                                                '{:.2%}'.format(v_neg_hold_win_times/v_neg_hold_times), 
                                                '{:.2f}'.format(v_neg_hold_period/v_neg_hold_times),
                              '{:.2%}'.format(v_hold_win_period/v_hold_period), 
                              '{:.2%}'.format(v_max_dd), 
                              '{:.2f}'.format(v_annual_ret/v_max_dd),
                              '{:.2%}'.format(v_annual_ret), 
                              '{:.2%}'.format(v_annual_std), 
                              '{:.2f}'.format(v_sharpe)
                             ]
    performance_df = pd.DataFrame(performance_values, index=performance_cols)
    
    return mkt_data, performance_df


###不设置止损

m15_300 = calc_vUvD(m15_300, nL=17)
m15_300 = calc_signal(m15_300)
m15_300['sl_price'] = None
m15_300 = calc_position(m15_300, shift_period=1, otime_limit=None, etime_limit=None, 
                          is_consider_open=True, is_consider_sl=False, is_daily_close=False,
                          comm=0.0002, leverage=None)

print(m15_300[m15_300['date']<='2011-12-09']['signal'].value_counts())


res_m15_300, performance_df = statistic_performance(m15_300[m15_300['date']<='2023-01-01'],
                                                    #m15_300[m15_300['date']>'2023-01-01'],
                                                    r0=0.03, 
                                                    data_period=15, 
                                                    is_consider_sl=False, 
                                                    )

print(performance_df)


###设置止损
m15_300 = calc_vUvD(m15_300, nL=15)
m15_300 = calc_signal(m15_300)
m15_300 = calc_sl_price(m15_300, default_af=0 , stepsize_af=0.01, max_af=0.1)
m15_300 = calc_position(m15_300, shift_period=1, otime_limit=None, etime_limit=None, 
                        is_consider_open=True, is_consider_sl=True, is_daily_close=False,
                        #comm=0.0002, 
                        leverage=None)

print(m15_300[m15_300['date']<='2023-01-01']['signal'].value_counts())


res_m15_300, performance_df = statistic_performance(#m15_300[m15_300['date']<='2023-01-01'],
                                                    m15_300[m15_300['date']>'2023-01-01'], 
                                                    r0=0.03, 
                                                    data_period=15, 
                                                    is_consider_sl=True, 
                                                    )

print(performance_df)

#nL参数稳定性
""" nL参数范围【2,50】 """  
nls = [i for i in range(2, 51, 1)]

""" 滚动计算 """
res_df = []
for nl in tqdm(nls):
    m15_300 = calc_vUvD(m15_300, nL=nl)
    m15_300 = calc_signal(m15_300)
    m15_300 = calc_sl_price(m15_300, default_af=0 , stepsize_af=0.01, max_af=0.1)
    m15_300 = calc_position(m15_300, shift_period=1, otime_limit=None, etime_limit=None, 
                            is_consider_open=True, is_consider_sl=True, is_daily_close=False,
                            comm=0.0002, leverage=None)
    res_m15_300, performance_df = statistic_performance(m15_300[m15_300['date']<='2023-01-01'],
                                                    r0=0.03, 
                                                    data_period=15, 
                                                    is_consider_sl=True, 
                                                    )
    annualr_2_maxdd = float(performance_df.values[16][0].strip('%'))/float(performance_df.values[14][0].strip('%'))
    cumu_r = float(performance_df.values[0][0].strip('%'))/100
    res_df.append([nl, annualr_2_maxdd, cumu_r])

""" 结果DataFrame """
res_df = pd.DataFrame(res_df, columns=['nL', '年化收益/最大回撤', '累计收益']).set_index('nL')
res_df.sort_values('累计收益', ascending=False)





import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 


res_df[['累计收益']].plot(figsize=(10,6), ylim=(0,1))
res_df[['年化收益/最大回撤']].plot(figsize=(10,6))



















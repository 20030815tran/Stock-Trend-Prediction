import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 
import seaborn as sns
import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from vnstock3 import Vnstock
from keras.models import load_model 


tickers = {
    "HOSE": ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG'],
    "VN30": ['VHM', 'VCB', 'VIC', 'MSN', 'GVR', 'TPB', 'STB', 'BID', 'FPT', 'ACB'],
    "VNMidCap": ['MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'TCB', 'TPB', 'VPB'],
    "VNSmallCap": ['VJC', 'VNM', 'HPG', 'HDB', 'ACB'],
    "VNAllShare": ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
                   'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
                   'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'],
    "VN100": ['VHM', 'VCB', 'VIC', 'MSN', 'GVR', 'TPB', 'STB', 'BID', 'FPT', 'ACB'],
    "ETF": ['VFMVN30', 'FUEVFVND', 'FUEVFVND'],
    "HNX": ['AAM', 'VND', 'HUT', 'SHS'],
    "HNX30": ['PVI', 'SHS', 'VND'],
    "HNXCon": ['VND', 'SHS'],
    "HNXFin": ['VND', 'SHS'],
    "HNXLCap": ['VND', 'SHS'],
    "HNXMSCap": ['VND', 'SHS'],
    "HNXMan": ['VND', 'SHS'],
    "UPCOM": ['VCG', 'VIG', 'UPH'],
    "FU_INDEX": ['FUESSV50', 'FUESSV30']
}

st.title( ' Stock Trend Prediction')

classification = st.selectbox('Select Stock Classification', list(tickers.keys()))
user_input = st.selectbox('Select Stock Ticker', tickers[classification])
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Start Date', value=pd.to_datetime('2018-01-01'))

with col2:
    end_date = st.date_input('End Date', value=pd.to_datetime('2022-12-31'))
    
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

stock = Vnstock().stock(symbol=user_input, source='VCI')
# df = stock.quote.history(start='2018-01-01', end='2022-12-31')
df = stock.quote.history(start=start_date_str, end=end_date_str)
df.set_index('time', inplace=True)

# st.subheader('Data from 2018 - 2022')
st.subheader('Data from {} to {}'.format(start_date, end_date))
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200,'g')
plt.plot(df.close, 'b')
st.pyplot(fig)

#Tương quan giá đóng cửa và khối lượng giao dịch
st.subheader('Correlation of closing price and trading volume')
plt.figure(figsize=(12, 6))
sct = sns.scatterplot(data=df, x='close', y='volume', color='green')
plt.xlabel('Closing Price')
plt.ylabel('Volume')
st.pyplot(plt)


#Thống kê lợi nhuận trung bình theo ngày qua các tháng trong năm (%)
st.subheader('Correlation of closing price and trading volume')
df['returns'] = df['close'].pct_change() * 100
return_pivot = pd.pivot_table(df, index=df.index.year, columns=df.index.month, values='returns', aggfunc='mean')
cmap = df.viz.create_cmap('percentage') 
return_pivot.viz.heatmap(figsize=(10, 6),
                         title='Average daily profit statistics over months of the year (%)',
                         annot=True,
                         cmap=cmap)
st.pyplot(plt)

#tỷ suất sinh lợi hàng ngày (Daily Percentage Returns) dưới dạng phần trăm
st.subheader('Daily Percentage Returns')
returns = 100 * df['close'].pct_change().dropna()
plt.figure(figsize=(12,6))
plt.plot(returns)
plt.ylabel('Pct Return', fontsize=16)
st.pyplot(plt)

st.subheader('partial autocorrelation')
plot_pacf(returns**2)
plt.show()
st.pyplot(plt)

#Fit GARCH(3,3)
model = arch_model(returns, p=3, q=3)
model_fit = model.fit()
model_fit.summary()
#Try GARCH(3,0) = ARCH(3) 
model = arch_model(returns, p=3, q=0)
model_fit = model.fit()
model_fit.summary()

rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])
st.subheader('Volatility Prediction - Rolling Forecast')
plt.figure(figsize=(12,8))
true, = plt.plot(returns[-365:])
preds, = plt.plot(rolling_predictions)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)    
st.pyplot(plt)

# model = arch_model(returns, p=2, q=2)
# model_fit = model.fit()
# model_fit.summary()

st.subheader('Volatility Prediction')
train = returns
model = arch_model(train, p=2, q=2)
model_fit = model.fit(disp='off')
pred = model_fit.forecast(horizon=7)
future_dates = [returns.index[-1] + timedelta(days=i) for i in range(1,8)]
pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)
plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Volatility Prediction - Next 7 Days', fontsize=20)
st.pyplot(plt)
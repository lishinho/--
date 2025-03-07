import pandas as pd
import pandas_ta as ta
import akshare as ak
from datetime import datetime, timedelta

# ========== 数据获取模块 ==========
def fetch_stock_data(symbol, start_date, end_date):
    """通过AKShare获取股票历史数据（日线）"""
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date)
        df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

# ========== 指标计算模块（使用 pandas_ta）==========
def calculate_indicators(df):
    """计算技术指标：均线、MACD、RSI、BOLL、成交量"""
    # 均线 (5日, 20日)
    df['ma5'] = df.ta.sma(length=5)
    df['ma20'] = df.ta.sma(length=20)
    
    # MACD (默认参数：fast=12, slow=26, signal=9)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # RSI (14日)
    df['rsi'] = df.ta.rsi(length=14)
    
    # BOLL (20日)
    boll = df.ta.bbands(length=20)
    df = pd.concat([df, boll], axis=1)
    
    # 成交量变化率（3日平均）
    df['volume_ma3'] = df['volume'].rolling(window=3).mean()
    df['volume_pct_change'] = (df['volume'] / df['volume_ma3'].shift(1)) - 1
    
    # 清理列名
    df.rename(columns={
        'MACD_12_26_9': 'macd',
        'MACDs_12_26_9': 'macd_signal',
        'MACDh_12_26_9': 'macd_hist',
        'BBL_20_2.0': 'boll_lower',
        'BBM_20_2.0': 'boll_mid',
        'BBU_20_2.0': 'boll_upper'
    }, inplace=True)
    
    return df

# ========== 信号生成模块 ==========
def generate_signals(df):
    """根据策略生成买卖信号"""
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0  # 0: 无信号, 1: 买入, -1: 卖出
    
    # 买入条件（至少满足2个）
    buy_condition = (
        (df['ma5'] > df['ma20']) &  # 均线金叉
        (df['macd'] > df['macd_signal']) &  # MACD金叉
        (df['rsi'] < 30) &  # RSI超卖
        (df['close'] < df['boll_lower']) &  # 股价触及BOLL下轨
        (df['volume_pct_change'] > 0.2)  # 成交量放大20%
    )
    
    # 卖出条件（任一条件触发）
    sell_condition = (
        (df['macd'] < df['macd_signal']) |  # MACD死叉
        (df['rsi'] > 70) |  # RSI超买
        (df['close'] > df['boll_upper'])  # BOLL触及上轨
    )
    
    signals.loc[buy_condition, 'signal'] = 1
    signals.loc[sell_condition, 'signal'] = -1
    return signals

# ========== 回测模块 ==========
def backtest_strategy(df, signals):
    """模拟交易回测"""
    df['position'] = signals['signal'].shift(1)  # 次日开盘执行
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

# ========== 主程序 ==========
if __name__ == "__main__":
    # 参数设置，支持多只股票
    symbols = ["000001", "600900","000651","601318","000977","000538","601995","600036","601088","002304"]  # 可以添加更多股票代码
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    for symbol in symbols:
        # 获取股票名称
        stock_info = ak.stock_individual_info_em(symbol)
        try:
            stock_name = stock_info['value'][stock_info['item'] == '股票名称'].values[0]
        except IndexError:
            print(f"无法获取 {symbol} 的股票名称，将使用空名称继续处理。")
            stock_name = ""

        # 获取数据
        df = fetch_stock_data(symbol, start_date, end_date)
        if df is None:
            continue

        # 计算指标并生成信号
        df = calculate_indicators(df)
        signals = generate_signals(df)

        # 回测并输出结果
        df = backtest_strategy(df, signals)

        print(f"===== {stock_name}({symbol}) 最新信号 =====")
        latest_date = df.index[-1].strftime('%Y-%m-%d')
        latest_signal = signals.iloc[-1]['signal']
        signal_text = '买入' if latest_signal == 1 else '卖出' if latest_signal == -1 else '无'
        latest_price = df['close'].iloc[-1]
        print(f"日期: {latest_date}, 信号: {signal_text}, 最新股价: {latest_price:.2f}")

        # 输出判定依据
        print("===== 判定依据 =====")
        latest_row = df.iloc[-1]
        if latest_signal == 1:
            conditions = [
                latest_row['ma5'] > latest_row['ma20'],
                latest_row['macd'] > latest_row['macd_signal'],
                latest_row['rsi'] < 30,
                latest_row['close'] < latest_row['boll_lower'],
                latest_row['volume_pct_change'] > 0.2
            ]
            condition_names = [
                "均线金叉",
                "MACD金叉",
                "RSI超卖",
                "股价触及BOLL下轨",
                "成交量放大20%"
            ]
            satisfied_conditions = [name for cond, name in zip(conditions, condition_names) if cond]
            print(f"买入依据: {', '.join(satisfied_conditions)}")
        elif latest_signal == -1:
            conditions = [
                latest_row['macd'] < latest_row['macd_signal'],
                latest_row['rsi'] > 70,
                latest_row['close'] > latest_row['boll_upper']
            ]
            condition_names = [
                "MACD死叉",
                "RSI超买",
                "BOLL触及上轨"
            ]
            satisfied_conditions = [name for cond, name in zip(conditions, condition_names) if cond]
            print(f"卖出依据: {', '.join(satisfied_conditions)}")
        else:
            print("无信号，不满足买卖条件。")

        # 输出股票基本信息
        print("===== 股票基本信息 =====")
        print(f"股票名称: {stock_name}")
        print(f"股票代码: {symbol}")
        print(f"数据起始日期: {start_date}")
        print(f"数据结束日期: {end_date}")

        print("===== 累计收益率 =====")
        print(f"{df['cum_returns'].iloc[-1]:.2%}")

        # 可视化（可选）
        df[['close', 'ma5', 'ma20', 'boll_upper', 'boll_lower']].plot(figsize=(12, 6), title=f"{stock_name} 价格与指标")
        df['cum_returns'].plot(figsize=(12, 4), title=f"{stock_name} 策略累计收益")
import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==================== 1. 页面配置 ====================
st.set_page_config(
    page_title="Pollutant Degradation-Persulfate Generation Prediction",
    page_icon="🧪",
    layout="wide"
)

# ==================== 2. 加载新模型资源 ====================
# 💡 核心修改：动态获取当前 web2.0.py 所在的文件夹路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 因为模型和代码在同一个目录下，所以模型目录就是当前目录！
MODEL_DIR = CURRENT_DIR


@st.cache_resource
def load_resources():
    try:
        # 直接在当前文件夹下读取
        scaler_X = joblib.load(os.path.join(MODEL_DIR, 'scaler_X.pkl'))
        model_rate = joblib.load(os.path.join(MODEL_DIR, 'model_污染物去除速率.pkl'))
        model_eff = joblib.load(os.path.join(MODEL_DIR, 'model_污染物降解效率.pkl'))
        model_energy = joblib.load(os.path.join(MODEL_DIR, 'model_能耗.pkl'))
        model_yield = joblib.load(os.path.join(MODEL_DIR, 'model_过硫酸盐产率.pkl'))

    except FileNotFoundError as e:
        st.error(f"❌ Error: 找不到模型文件。\n系统实际去寻找的路径是: {e.filename}")
        st.stop()

    return scaler_X, model_rate, model_eff, model_energy, model_yield


# 提取加载好的资源
scaler_X, model_rate, model_eff, model_energy, model_yield = load_resources()

# ==================== 3. 定义输入输出名称 ====================
input_names_en = [
    "Pollutant Concentration (μmol/L)",
    "Sulfate Concentration (mmol/L)",
    "Reaction pH Value",
    "Applied Voltage (V vs. SCE)"
]

output_names_en = [
    "Reaction Rate Constant (min⁻¹)",  # 对应 污染物去除速率
    "Degradation Efficiency (%)",  # 对应 污染物降解效率
    "Persulfate Yield Rate (µM cm⁻² h⁻¹)",  # 对应 过硫酸盐产率
    "Energy Consumption EE/O (kW·hm⁻³)"  # 对应 能耗
]

# ==================== 4. 侧边栏 ====================
st.sidebar.header("⚙️ Parameters")
st.sidebar.markdown("Adjust experimental conditions:")


def user_input_features():
    input_1 = st.sidebar.number_input(f"{input_names_en[0]}", min_value=0.0, value=20.0, step=0.5)
    input_2 = st.sidebar.number_input(f"{input_names_en[1]}", min_value=0.0, value=50.0, step=0.1)
    input_3 = st.sidebar.slider(f"{input_names_en[2]}", min_value=1.0, max_value=12.0, value=7.0, step=0.1)
    input_4 = st.sidebar.number_input(f"{input_names_en[3]}", min_value=0.0, value=3.0, step=0.1)

    return np.array([[input_1, input_2, input_3, input_4]])


input_data = user_input_features()

# ==================== 5. 主页面 ====================
st.title("Pollutant Degradation-Persulfate Generation Prediction")
st.markdown("---")

# 显示当前输入
st.subheader("Current Input Conditions")
st.dataframe(pd.DataFrame(input_data, columns=input_names_en), hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# 预测按钮
if st.button("PREDICT", type="primary", use_container_width=True):

    with st.spinner('Calculating...'):
        # 1. 对输入特征进行标准化
        input_scaled = scaler_X.transform(input_data)

        # 2. 四大模型分别独立推理
        # [去除速率] - 特殊处理：模型输出的是对数值，需要还原
        pred_rate_log = model_rate.predict(input_scaled)
        rem_rate = (10 ** pred_rate_log[0]) - 1e-6

        # [降解效率]
        deg_eff = model_eff.predict(input_scaled)[0]

        # [能耗]
        energy = model_energy.predict(input_scaled)[0]

        # [过硫酸盐产率]
        yield_rate = model_yield.predict(input_scaled)[0]

    # ==================== 6. 结果展示 ====================
    st.markdown("### Prediction Results")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    # 按照前端 UI 期望的顺序填入对应数值
    with col1:
        st.metric(label=output_names_en[0], value=f"{rem_rate:.6f}")
    with col2:
        st.metric(label=output_names_en[1], value=f"{deg_eff:.2f}")
    with col3:
        st.metric(label=output_names_en[2], value=f"{yield_rate:.2f}")
    with col4:
        st.metric(label=output_names_en[3], value=f"{energy:.4f}")

    st.markdown("---")

    # 英文提示信息 (根据实际能耗情况评估)
    if energy > 5.0:
        st.warning(f"⚠️ **Note**: Predicted energy consumption is high ({energy:.3f}). Optimization recommended.")
    else:
        st.success(f"✅ **Good**: Energy consumption is within a reasonable range.")

else:
    st.info("👈 Please adjust parameters in the sidebar and click the button to predict.")

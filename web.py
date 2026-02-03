import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# ==================== 1. 页面配置 ====================
st.set_page_config(
    page_title="Pollutant Treatment Prediction",
    page_icon="🧪",
    layout="wide"
)


# ==================== 2. 模型结构 (保持不变) ====================
class PollutionNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


# ==================== 3. 加载资源 ====================
@st.cache_resource
def load_resources():
    try:
        scalers = joblib.load('pollution_scalers.pkl')
    except FileNotFoundError:
        st.error("❌ Error: 'pollution_scalers.pkl' not found.")
        st.stop()

    model = PollutionNet()
    try:
        model.load_state_dict(torch.load('pollution_model_weights.pth', map_location=torch.device('cpu')))
        model.eval()
    except FileNotFoundError:
        st.error("❌ Error: 'pollution_model_weights.pth' not found.")
        st.stop()

    return model, scalers


model, scalers = load_resources()
scaler_X = scalers['scaler_X']
scaler_y = scalers['scaler_y']

# 【核心修改点 1】: 手动定义英文的输入和输出名称，覆盖原本的中文
input_names_en = [r"Pollutant Concentration (mg/L)", r"Sulfate Concentration (mmol/L)", r"pH Value", r"Voltage (V vs. SCE)"]
output_names_en = [
    r"Removal Rate (min$^{-1}$)",  # 对应 污染物去除速率
    r"Degradation Efficiency (%)",  # 对应 污染物降解效率
    r"Persulfate Yield Rate (µM cm$^{-2}$ h$^{-1}$)",  # 对应 过硫酸盐产率
    r"Energy Consumption EE/O (kW•hm$^{-3}$)"  # 对应 能耗
]

# ==================== 4. 侧边栏 (已替换为你写的英文版) ====================
st.sidebar.header("⚙️ Parameters")
st.sidebar.markdown("Adjust experimental conditions:")


def user_input_features():
    # 使用你提供的代码，并将 input_names_en 用于对应的标签
    input_1 = st.sidebar.number_input(f"{input_names_en[0]}", min_value=0.0, value=20.0, step=0.5)
    input_2 = st.sidebar.number_input(f"{input_names_en[1]}", min_value=0.0, value=50.0, step=0.1)
    input_3 = st.sidebar.slider(f"{input_names_en[2]}", min_value=1.0, max_value=12.0, value=7.0, step=0.1)
    input_4 = st.sidebar.number_input(f"{input_names_en[3]}", min_value=0.0, value=1.5, step=0.1)

    return np.array([[input_1, input_2, input_3, input_4]])


input_data = user_input_features()

# ==================== 5. 主页面 ====================
st.title("Pollutant Degradation Prediction")
st.markdown("---")

# 显示当前输入
st.subheader("Current Input Conditions")
# 为了表格好看，这里也用英文列名
st.dataframe(pd.DataFrame(input_data, columns=input_names_en), hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# 预测按钮
if st.button("PREDICT", type="primary", use_container_width=True):

    with st.spinner('Calculating...'):
        # 1. 归一化
        input_scaled = scaler_X.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        # 2. 推理
        with torch.no_grad():
            prediction_scaled = model(input_tensor)

        # 3. 反归一化
        prediction = scaler_y.inverse_transform(prediction_scaled.numpy())
        res = prediction[0]

    # 结果展示
    st.markdown("### Prediction Results")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    # 【核心修改点 2】: label 使用 output_names_en
    with col1:
        st.metric(label=output_names_en[0], value=f"{res[0]:.4f}")
    with col2:
        st.metric(label=output_names_en[1], value=f"{res[1]:.2f}")
    with col3:
        st.metric(label=output_names_en[2], value=f"{res[2]:.2f}")
    with col4:
        st.metric(label=output_names_en[3], value=f"{res[3]:.4f}")

    st.markdown("---")
    # 英文提示信息
    if res[3] > 5.0:
        st.warning(f"⚠️ **Note**: Predicted energy consumption is high ({res[3]:.3f}). Optimization recommended.")
    else:
        st.success(f"✅ **Good**: Energy consumption is within a reasonable range.")

else:
    st.info("👈 Please adjust parameters in the sidebar and click the button to predict.")
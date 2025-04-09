# streamlit_robot_app_plotly_zh_v2.py
import sys
import numpy as np
import streamlit as st
import time
import plotly.graph_objects as go # 导入 Plotly

# --- 几何常量、计算常量和辅助函数 (保持不变, 除了移除 static_line 定义) ---
top_vertex = np.array([0, 5]); bottom_left_corner = np.array([-3, 0])
rect_inner_bl = np.array([-2, 0]); rect_outer_bl = np.array([-2, -5])
rect_outer_br = np.array([2, -5]); rect_inner_br = np.array([2, 0])
bottom_right_corner = np.array([3, 0]); upper_right_vertex = np.array([5, 3])
upper_left_vertex = np.array([-5, 3]); dot_line_left = np.array([-4, 4])
dot_line_right = np.array([4, 4]); vec_line1 = np.array([[-5/2], [4]]) # 左耳曲线起点
vec_line2 = np.array([[5/2], [4]]) # 右耳曲线起点
line_left_alpha = np.arctan(2/5)
line_right_alpha = np.arctan(-2/5)
matrix_trans_left = np.array([[np.cos(line_left_alpha), -np.sin(line_left_alpha)], [np.sin(line_left_alpha), np.cos(line_left_alpha)]])
matrix_trans_right = np.array([[np.cos(line_right_alpha), -np.sin(line_right_alpha)], [np.sin(line_right_alpha), np.cos(line_right_alpha)]])
L = np.linalg.norm(dot_line_right - dot_line_left); L = max(L, 1e-6) # 耳朵曲线基准弦长
polygon_vertices_base = np.array([ bottom_left_corner, rect_inner_bl, rect_outer_bl, rect_outer_br, rect_inner_br, bottom_right_corner, upper_right_vertex, top_vertex, upper_left_vertex, bottom_left_corner ]) # 机器人主体
# --- 移除不再需要的 static_line 定义 ---
# static_line1_base = np.array([[-5, -2.5], [3, 4]])
# static_line2_base = np.array([[2.5, 5], [4, 3]])
beta_epsilon = 1e-7; rotation_center = np.array([0, 0]) # 旋转中心

def rotate_points(points_2xn, angle_rad, center_2x1):
    """围绕指定中心旋转点集 (2xN 数组)。"""
    if points_2xn is None or points_2xn.size == 0: return points_2xn
    center = center_2x1.reshape(2, 1)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    translated_points = points_2xn - center; rotated_points = rotation_matrix @ translated_points
    return rotated_points + center

# --- 修正后的 get_curve_points 函数 (用于计算耳朵曲线) ---
def get_curve_points(beta, num_pts=80):
    """根据 beta 值计算左右两条“耳朵”曲线的点坐标。"""
    if np.isclose(beta, 0, atol=beta_epsilon):
        t_line = np.linspace(0, 1, max(num_pts, 2)).reshape(1, -1)
        points = vec_line1 + (vec_line2 - vec_line1) * t_line
        return np.copy(points), np.copy(points) # 返回直线段作为耳朵
    safe_beta = np.sign(beta) * max(abs(beta), 1e-10)
    radius = L / (2 * safe_beta)
    max_radius = 1e6
    radius = np.clip(radius, -max_radius, max_radius)
    t_curve = np.linspace(0, 2 * safe_beta, num_pts)
    line_x0 = (1 - np.cos(t_curve)) * radius
    line_y0 = np.sin(t_curve) * radius
    line_xy0 = np.array([line_x0, line_y0])
    line_left_xy = matrix_trans_left @ line_xy0 + vec_line1
    line_right_xy = matrix_trans_right @ line_xy0 + vec_line2
    line_left_xy = np.nan_to_num(line_left_xy)
    line_right_xy = np.nan_to_num(line_right_xy)
    # --- 省略维度检查和重塑代码 (假设输入输出正常) ---
    # ... (之前的维度检查代码) ... # 为简洁省略，但实际应用中保留是好的
    # 确保返回 2xN
    if line_left_xy.ndim == 1: line_left_xy = line_left_xy.reshape(2, -1)
    if line_right_xy.ndim == 1: line_right_xy = line_right_xy.reshape(2, -1)
    if line_left_xy.shape[0]!=2: # 简单转置处理常见错误
        if line_left_xy.shape[1]==2: line_left_xy=line_left_xy.T
        else: line_left_xy = np.array([[],[]]) # 无法处理则返回空
    if line_right_xy.shape[0]!=2:
        if line_right_xy.shape[1]==2: line_right_xy=line_right_xy.T
        else: line_right_xy = np.array([[],[]])

    return line_left_xy, line_right_xy


# --- 更新后的 Plotly 绘图函数 ---
def plot_robot_plotly(beta, theta, x_lim, y_lim):
    """
    使用 Plotly 生成机器人姿态图，"耳朵"指红/紫色曲线。
    返回一个 plotly.graph_objects.Figure 对象。
    """
    fig = go.Figure() # 创建 Plotly 图形对象

    # --- 1. 计算几何数据 ---
    try:
        # 计算 "耳朵" 曲线
        curve_l_base, curve_r_base = get_curve_points(beta)
    except Exception as e:
        print(f"计算耳朵曲线时出错 β={beta}: {e}")
        curve_l_base, curve_r_base = np.array([[],[]]), np.array([[],[]]) # 出错时返回空数组

    # 旋转主体和 "耳朵" 曲线
    poly_rot = rotate_points(polygon_vertices_base.T, theta, rotation_center)
    curve_l_rot = rotate_points(curve_l_base, theta, rotation_center) # 旋转后的左耳
    curve_r_rot = rotate_points(curve_r_base, theta, rotation_center) # 旋转后的右耳
    # --- 不再计算 static lines ---
    # static_line1_rot = ...
    # static_line2_rot = ...

    # --- 2. 添加绘图轨迹 (Trace) 到 Plotly 图形 ---

    # 绘制主体多边形 (填充)
    if poly_rot.shape[1] > 0:
        fig.add_trace(go.Scatter(
            x=poly_rot[0, :], y=poly_rot[1, :],
            mode='lines',
            fill='toself',
            fillcolor='rgba(135, 206, 250, 0.6)',
            line=dict(color='rgba(0, 0, 139, 0.9)', width=2),
            name='主体' # 图例名称
        ))

    # 绘制左耳 (红色曲线)
    if curve_l_rot.shape[1] > 1:
        fig.add_trace(go.Scatter(
            x=curve_l_rot[0, :], y=curve_l_rot[1, :],
            mode='lines',
            line=dict(color='red', width=3),
            name='左耳' # 更新图例名称
        ))

    # 绘制右耳 (紫色曲线)
    if curve_r_rot.shape[1] > 1:
        fig.add_trace(go.Scatter(
            x=curve_r_rot[0, :], y=curve_r_rot[1, :],
            mode='lines',
            line=dict(color='purple', width=3),
            name='右耳' # 更新图例名称
        ))

    # --- 不再绘制绿色虚线 ---
    # fig.add_trace(go.Scatter(... static_line1_rot ...))
    # fig.add_trace(go.Scatter(... static_line2_rot ...))

    # --- 3. 配置图形布局 ---
    fig.update_layout(
        title=f'机器人状态 (β={beta:.3f}, θ={theta:.3f})', # 图形标题
        xaxis=dict(
            range=x_lim, # 设置 X 轴范围
            showgrid=True, # 显示网格
            gridcolor='rgba(128,128,128,0.5)'
        ),
        yaxis=dict(
            range=y_lim, # 设置 Y 轴范围
            scaleanchor="x",  # 保持 X/Y 轴比例一致
            scaleratio=1,
            showgrid=True, # 显示网格
            gridcolor='rgba(128,128,128,0.5)'
        ),
        template='plotly_dark', # 使用 Plotly 的深色主题
        showlegend=True, # 显示图例
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99), # 图例位置
        margin=dict(l=20, r=20, t=50, b=20), # 调整边距
    )

    return fig # 返回配置好的 Plotly 图形对象


# --- Streamlit 应用主逻辑 ---
st.set_page_config(page_title="机器人运动模拟 (Plotly)", layout="wide")
st.title("🤖 机器人运动模拟 (Plotly版)")
st.caption("通过侧边栏控件交互式探索机器人姿态（红色/紫色曲线代表耳朵）。") # 更新说明

# 初始化 Session State
if 'sensor_mode' not in st.session_state: st.session_state.sensor_mode = False
if 'beta_manual' not in st.session_state: st.session_state.beta_manual = 0.0
if 'theta_manual' not in st.session_state: st.session_state.theta_manual = 0.0

# 计算绘图范围 (使用缓存，移除 static line 相关计算)
@st.cache_data
def calculate_plot_limits():
    """计算绘图范围以包含所有可能的姿态。"""
    padding = 5.0; betas_to_check = np.linspace(-np.pi/2 * 0.95, np.pi/2 * 0.95, 5)
    all_x = polygon_vertices_base[:, 0].copy(); all_y = polygon_vertices_base[:, 1].copy()
    for beta_check in betas_to_check:
        thetas_to_check = np.linspace(-abs(beta_check)*2, abs(beta_check)*2, 5)
        try: curve_l_base, curve_r_base = get_curve_points(beta_check, num_pts=20) # 用少量点计算范围
        except Exception: curve_l_base, curve_r_base = np.array([]), np.array([])
        for theta_check in thetas_to_check:
            poly_rot = rotate_points(polygon_vertices_base.T, theta_check, rotation_center)
            all_x = np.concatenate((all_x, poly_rot[0, :])); all_y = np.concatenate((all_y, poly_rot[1, :]))
            # 只考虑主体和耳朵曲线的范围
            if curve_l_base.size > 0: curve_l_rot = rotate_points(curve_l_base, theta_check, rotation_center); all_x = np.concatenate((all_x, curve_l_rot[0, :])); all_y = np.concatenate((all_y, curve_l_rot[1, :]))
            if curve_r_base.size > 0: curve_r_rot = rotate_points(curve_r_base, theta_check, rotation_center); all_x = np.concatenate((all_x, curve_r_rot[0, :])); all_y = np.concatenate((all_y, curve_r_rot[1, :]))
            # --- 不再包含 static lines 的范围 ---
            # line1_rot = ...; line2_rot = ...
            # all_x = np.concatenate((all_x, line1_rot[0, :], line2_rot[0, :])); ...
    x_min, x_max = -15, 15; y_min, y_max = -15, 15
    valid_x = all_x[~np.isnan(all_x)]; valid_y = all_y[~np.isnan(all_y)]
    if valid_x.size > 0: x_min = np.min(valid_x) - padding; x_max = np.max(valid_x) + padding
    if valid_y.size > 0: y_min = np.min(valid_y) - padding; y_max = np.max(valid_y) + padding + 2.0
    x_lim = (min(x_min, -15), max(x_max, 15)); y_lim = (min(y_min, -15), max(y_max, 15))
    return x_lim, y_lim

x_lim, y_lim = calculate_plot_limits() # 获取绘图范围

# --- 侧边栏控件 (逻辑不变) ---
with st.sidebar:
    st.header("模拟参数")
    st.session_state.sensor_mode = st.checkbox(
        "使用传感器模式 (控制 θ)",
        value=st.session_state.sensor_mode,
        key="cb_mode"
    )
    current_beta = 0.0; current_theta = 0.0
    if st.session_state.sensor_mode:
        st.write("在此模式下，θ 由滑块控制, 且 θ = 2 * β。")
        st.session_state.theta_manual = st.slider(
            "θ (弧度)", -np.pi, np.pi, st.session_state.theta_manual, 0.01, key="slider_theta",
            help="拖动滑块改变整体旋转角度 θ。耳朵形态 β 会随之改变。" # 更新提示信息
        )
        current_theta = st.session_state.theta_manual; current_beta = current_theta / 2.0
        st.metric("计算得到的 β", f"{current_beta:.3f}"); st.caption(f"(当前 θ = {current_theta:.3f})")
    else:
        st.write("在此模式下，β 由滑块控制（改变耳朵形态），θ 固定为 0。") # 更新说明
        st.session_state.beta_manual = st.slider(
            "β (弧度)", -np.pi/2 * 0.99, np.pi/2 * 0.99, st.session_state.beta_manual, 0.01, key="slider_beta",
            help="拖动滑块改变耳朵形态 β。机器人整体不旋转 (θ=0)。" # 更新提示信息
        )
        current_beta = st.session_state.beta_manual; current_theta = 0.0
        st.metric("固定 θ", f"{current_theta:.3f}"); st.caption(f"(当前 β = {current_beta:.3f})")
    st.markdown("---")
    st.caption("提示：拖动上面的滑块查看不同姿态。")

# --- 主绘图区域 ---
st.subheader("机器人姿态可视化 (Plotly 渲染)")

# --- 生成 Plotly 图形 ---
fig_plotly = plot_robot_plotly(current_beta, current_theta, x_lim, y_lim)

# --- 在 Streamlit 中显示 Plotly 图形 ---
st.plotly_chart(fig_plotly, use_container_width=True)

# --- 侧边栏底部信息 ---
st.sidebar.markdown("---")
st.sidebar.info("使用 Plotly 绘图的机器人模拟应用。")
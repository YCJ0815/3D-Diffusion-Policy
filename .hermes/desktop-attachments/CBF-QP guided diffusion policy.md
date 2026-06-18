你现在需要在现有 diffusion policy 轨迹生成代码中，实现一个“Late-stage Surface-sample CBF-QP Guided Denoising”模块。

一、当前任务背景

当前模型用于机械臂焊接过渡段路径规划。Diffusion policy 的输出不是 dense waypoint，而是 B-spline 控制点残差。模型一次生成多条候选轨迹，每条轨迹通过 B-spline 控制点恢复为关节轨迹。

当前希望在 diffusion 去噪的最后 5–10 步中，利用机械臂表面采样点构建 CBF-QP 安全投影，使当前 clean control-point residual estimate 被投影到更安全的区域，然后再继续 DDIM/DDPM 去噪。最终输出前必须做高密度安全证书检查；若没有通过安全证书的候选，则进入已有 terminal CBF 后处理或 fallback planner。

注意：当前碰撞建模方式是“机械臂表面采样点”，不是碰撞球。因此安全函数不需要减球半径。

二、需要实现的主流程

实现函数：

    sample_with_surface_cbf_qp_guidance(obs, q_start, q_goal)

输入：
    obs: diffusion policy 的条件输入
    q_start: 起点关节角，shape [dof]
    q_goal: 终点关节角，shape [dof]

输出：
    C_best: 通过安全证书的 B-spline 控制点
    Q_best: 对应 dense joint trajectory
    info: 包含 collision score、QP success、certificate success、耗时等日志

整体流程：

1. 初始化 diffusion noisy residual:
       x_K ~ N(0, I)

   x 的含义是 normalized B-spline control-point residual:
       x ∈ R^{N_candidates × M × dof}

2. 按 DDIM/DDPM 从 k=K 到 k=1 去噪。

3. 每一步先通过 diffusion model 得到噪声预测:
       eps = model(x_k, k, obs)

4. 根据 eps 估计 clean residual:
       x0_hat = (x_k - sqrt(1 - alpha_bar[k]) * eps) / sqrt(alpha_bar[k])

   如果当前模型是 sample prediction，则直接使用模型输出作为 x0_hat。

5. 只有在最后 G 个 denoising steps 中启用 CBF-QP guidance。推荐：
       K_ddim = 10~20
       G_guidance = 5

6. 在 guidance step 中：
       a. 将 x0_hat 反归一化为控制点残差 ΔC
       b. 构造基准控制点 C_base
       c. 恢复完整控制点 C = C_base + ΔC
       d. 强制端点控制点等于 q_start 和 q_goal
       e. B-spline 插值得到 dense checking trajectory Q = B_check C
       f. 对 Q 中每个 waypoint 和机械臂表面采样点计算安全函数 h
       g. 选择 active constraints
       h. 对浅层碰撞候选解一个小规模 Surface-sample CBF-QP
       i. 得到 C_proj
       j. 将 C_proj 转回 normalized residual x0_proj
       k. 用 x0_proj 重新计算一致噪声 eps_proj
       l. 用 x0_proj 和 eps_proj 继续 DDIM 更新到 x_{k-1}

7. 完成 diffusion sampling 后，对所有候选做最终安全证书检查：
       Q_cert = B_cert C_final
       要求所有机械臂表面采样点满足:
           phi(x_l(q_t)) - d_safe >= d_cert
       并对相邻 waypoint 做 swept interpolation check。

8. 若存在通过安全证书的候选，选择综合评分最优的轨迹输出。

9. 若没有通过安全证书的候选，调用已有 terminal RS-CBF projection 或 fallback planner。

三、B-spline 控制点恢复

Diffusion 输出为 normalized control-point residual:

    x0_hat = ΔC_norm

反归一化：

    ΔC = ΔC_norm * deltaC_std + deltaC_mean

构造基准控制点：

    c_i_base = q_start + i / (M - 1) * (q_goal - q_start)

恢复控制点：

    C = C_base + ΔC

端点约束：

    C[0] = q_start
    C[-1] = q_goal

更稳定的版本固定前后两个控制点：

    C[0] = q_start
    C[1] = q_start
    C[-2] = q_goal
    C[-1] = q_goal

在 QP 中只允许优化中间控制点，例如 M=10 时优化 C[2:8]。

四、机械臂表面采样点安全函数

当前使用机械臂表面采样点，不使用碰撞球。

设第 l 个表面采样点在其所属 link 局部坐标系下为：

    x_bar_l ∈ R^3

其所属 link 为：

    link_id(l)

通过 FK 得到该点在世界坐标系下的位置：

    x_l(q) = FK_link_id(l)(q) @ x_bar_l

环境 SDF 为：

    phi(x)

其中：
    phi(x) > 0 表示点在障碍物外部
    phi(x) = 0 表示点在障碍物表面
    phi(x) < 0 表示点在障碍物内部

安全函数：

    h_l(q) = phi(x_l(q)) - d_safe

对 B-spline 第 t 个轨迹点：

    q_t = Σ_i B[t, i] * c_i

因此：

    h_{t,l}(C) = phi(x_l(q_t)) - d_safe

安全要求：

    h_{t,l}(C) >= 0

最终证书阶段要求：

    h_{t,l}(C) >= d_cert

五、表面采样点 Jacobian

需要实现或调用函数：

    compute_surface_point_jacobian(q, surface_sample_l)

返回：

    J_l(q) = ∂x_l(q) / ∂q

对于 6 自由度机械臂：

    J_l(q) ∈ R^{3 × 6}

其物理意义：

    x_dot_l = J_l(q) q_dot

对于 revolute joint j，如果该关节影响该表面点，则：

    J_l[:, j] = z_j(q) × (x_l(q) - o_j(q))

其中：
    z_j(q): 第 j 个关节轴在世界坐标系下的方向
    o_j(q): 第 j 个关节原点在世界坐标系下的位置

如果 joint j 不影响该 link，则：

    J_l[:, j] = 0

六、安全函数梯度

SDF 梯度：

    n_l(q) = ∇_x phi(x_l(q))

关节空间安全梯度：

    g_{t,l} = ∇_q h_l(q_t)
            = J_l(q_t)^T n_l(q_t)

其中：

    g_{t,l} ∈ R^{dof}

由于：

    q_t = Σ_i B[t, i] c_i

所以：

    ∂h_{t,l} / ∂c_i = B[t, i] * g_{t,l}

七、active constraints 选择

对每个候选轨迹，计算所有：

    h_{t,l} = phi(x_l(q_t)) - d_safe

选择危险约束集合：

    A = {(t,l) | h_{t,l} < d_trigger}

只保留最危险的 A_max 个：

    A_max = 16 或 32

排序依据：

    h_{t,l} 越小越危险

深度碰撞轨迹可以跳过 QP，避免浪费时间：

    if min(h) < -eps_deep:
        skip QP

推荐参数：

    d_safe = 0.02~0.04 m
    d_trigger = 0.05~0.08 m
    eps_deep = 0.03 m
    A_max = 16 或 32

八、Surface-sample CBF-QP

决策变量是自由控制点修正量：

    δC_free

向量化：

    z = vec(δC_free)

对每个 active constraint a=(t,l)，线性化安全函数：

    h_{t,l}(C + δC)
    ≈ h_{t,l}(C)
      + Σ_{i∈I_free} B[t,i] * g_{t,l}^T δc_i

加入 slack：

    h_{t,l}(C)
    + Σ_{i∈I_free} B[t,i] * g_{t,l}^T δc_i
    + ξ_{t,l}
    >= h_target(k)

其中：
    ξ_{t,l} >= 0

h_target(k) 在最后 G 个 denoising steps 中逐渐收紧。建议：

    第 1 个 guidance step: h_target = -0.02
    第 2 个 guidance step: h_target = -0.01
    第 3 个 guidance step: h_target = -0.005
    第 4 个 guidance step: h_target = 0
    第 5 个 guidance step: h_target = 0

QP 目标函数：

    minimize
        ||δC_free||^2
        + lambda_s ||D2(C + δC)||^2
        + rho ||ξ||^2

其中：
    D2 是控制点二阶差分矩阵
    lambda_s = 0.1~1.0
    rho = 1e4~1e6

约束包括：

1. 线性化 CBF 约束：
       h_{t,l}(C)
       + Σ_{i∈I_free} B[t,i] * g_{t,l}^T δc_i
       + ξ_{t,l}
       >= h_target(k)

2. slack 非负：
       ξ_{t,l} >= 0

3. trust region：
       -δ_max <= δc_i <= δ_max

   推荐：
       δ_max = 0.03~0.08 rad

4. 关节限位：
       q_min <= B_limit(C + δC) <= q_max

   可以只在较稀疏的 waypoint 上加该约束，避免 QP 过大。

5. 固定端点：
       δc_i = 0 for i not in I_free

QP 失败时，不要中断整个 diffusion sampling，直接返回原始 C。

九、重新注入 diffusion

QP 得到：

    C_proj = C + δC*

转回控制点残差：

    ΔC_proj = C_proj - C_base

归一化：

    x0_proj = (ΔC_proj - deltaC_mean) / deltaC_std

为了与当前 x_k 保持一致，重新计算噪声：

    eps_proj =
        (x_k - sqrt(alpha_bar[k]) * x0_proj)
        / sqrt(1 - alpha_bar[k])

确定性 DDIM 更新：

    x_{k-1}
    =
    sqrt(alpha_bar[k-1]) * x0_proj
    +
    sqrt(1 - alpha_bar[k-1]) * eps_proj

在最后 guidance 阶段建议：

    ddim_eta = 0

十、最终安全证书检查

最终恢复所有候选控制点：

    C_final = C_base + Denormalize(x_0)

高密度采样：

    Q_cert = B_cert C_final

推荐：

    T_cert = 256 或 512

对所有 waypoint 和表面采样点检查：

    h_{t,l} = phi(x_l(q_t)) - d_safe

要求：

    min_{t,l} h_{t,l} >= d_cert

推荐：

    d_cert = 0.005~0.015 m

还需要 swept check。对相邻 waypoint q_t 和 q_{t+1} 插入 S 个中间点：

    q_{t,β} = (1 - β) q_t + β q_{t+1}

其中：

    β ∈ {1/(S+1), 2/(S+1), ..., S/(S+1)}

推荐：

    S = 3~5

对所有中间点同样检查：

    phi(x_l(q_{t,β})) - d_safe >= d_cert

只有通过 certificate check 的轨迹才能输出。

十一、点云最近点版本兼容

如果暂时没有 SDF，可以使用障碍点云最近点替代。

对机械臂表面采样点 x_l(q)，找到最近障碍点：

    p_near = argmin_{p∈P_obs} ||x_l(q) - p||

距离：

    d_l(q) = ||x_l(q) - p_near||

方向：

    n_l(q) = (x_l(q) - p_near) / ||x_l(q) - p_near||

安全函数：

    h_l(q) = d_l(q) - d_safe

梯度：

    g_{t,l} = J_l(q_t)^T n_l(q_t)

其余 QP 公式不变。

十二、速度控制要求

不要对所有候选都解 QP。推荐策略：

1. 所有候选先快速计算 h_min。
2. 已经安全的候选跳过 QP。
3. 深度碰撞的候选跳过 QP。
4. 只对 shallow collision 或 near-obstacle 的 Top-K 候选解 QP。

推荐参数：

    N_candidates = 32
    K_ddim = 10~20
    G_guidance = 5
    K_qp_per_step = 4 或 8
    A_max = 16 或 32
    T_check = 64 或 128
    T_cert = 256 或 512

十三、需要记录的日志

每次规划需要记录：

    dp_time
    guidance_time
    qp_time
    certificate_time
    total_time

    h_min_before_guidance
    h_min_after_guidance
    h_min_final

    num_qp_called
    num_qp_success
    num_active_constraints

    certificate_success
    fallback_used

    goal_error
    smoothness
    path_length

十四、最小可实现版本

先实现以下最小版本：

    N_candidates = 32
    K_ddim = 10
    G_guidance = 5
    每步只对 Top-4 候选解 QP
    每个 QP active constraints = 16
    只优化中间控制点
    T_check = 64
    T_cert = 256
    swept interpolation = 3

目标是验证：

1. DP only 的碰撞率约为当前验证集 15%。
2. 加入 late-stage surface-sample CBF-QP guidance 后，最终碰撞率下降。
3. certificate success rate 上升。
4. 平均推理时间仍显著低于 RRT/TrajOpt。
5. fallback 比例较低。

请根据以上要求实现模块化代码，优先保证接口清晰、日志完整、QP 失败时系统不会崩溃，并确保最终输出前必须经过 certificate check。
# 光线追踪实验报告

## 一、实验目标

1. **理论理解**：理解光线投射与光线追踪的本质区别
2. **全局光照**：掌握通过次级射线实现硬阴影和理想镜面反射的方法
3. **GPU编程思维**：学习将递归光线追踪改写为适合GPU的迭代模式

## 二、实验原理

### 2.1 光线追踪基础

光线追踪的核心思想是从摄像机出发，向场景中发射光线，追踪光线与物体的交互过程。与光线投射只计算直接光照不同，光线追踪通过递归追踪反射、折射光线，能够模拟更丰富的光学效果。

### 2.2 Whitted风格光线追踪

本实验采用经典的Whitted模型，主要包含以下内容：

1. **主光线**：从摄像机出发，穿过像素平面，寻找场景中的第一个交点
2. **阴影射线**：从交点向光源发射，检测是否被遮挡
3. **反射射线**：在镜面材质表面，根据反射定律生成新的光线继续传播

### 2.3 关键技术

- **反射向量计算**：R = Lin - 2(Lin·N)N
- **阴影检测**：通过比较阴影射线与场景的交点距离来判断是否处于阴影中
- **GPU迭代实现**：用for循环替代递归，避免GPU栈溢出

## 三、实验内容与实现

### 3.1 场景搭建

使用Taichi在GPU上隐式定义了三个几何体：

1. **棋盘格地板**：y = -1.0的无限大平面，通过交点x和z坐标奇偶性生成纹理
2. **玻璃球**：位于(-1.5, 0.0, 0)，半径为1.0，呈现淡蓝色半透明效果
3. **镜面球**：位于(1.5, 0.0, 0)，半径为1.0，银色高反射材质

### 3.2 核心功能实现

#### 3.2.1 光线与物体求交

实现了球体和无限大平面的求交算法：

```python
@ti.func
def intersect_sphere(ro, rd, center, radius):
    # 计算判别式判断是否相交
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0.0:
        # 取最近的交点
        t = (-b - ti.sqrt(delta)) / 2.0
        if t > 0.0:
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal
```

#### 3.2.2 迭代光线追踪

使用for循环实现光线迭代弹射：

```python
for bounce in range(max_bounces[None]):
    t, N, obj_color, mat_id = scene_intersect(current_ro, current_rd)
    if t > 1e9:  # 未击中任何物体
        final_color += throughput * bg_color
        break
    # 根据材质类型处理...
```

#### 3.2.3 玻璃材质

实现了基于斯涅尔定律的折射效果：

```python
@ti.func
def refract(I, N, eta):
    cos_theta_i = ti.math.clamp(I.dot(N), -1.0, 1.0)
    sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i
    sin_theta_t_sq = eta * eta * sin_theta_i_sq
    
    if sin_theta_t_sq <= 1.0:
        # 计算折射方向
        cos_theta_t = ti.sqrt(1.0 - sin_theta_t_sq)
        if cos_theta_i < 0.0:
            refracted = -I * eta + N * (eta * cos_theta_i - cos_theta_t)
        else:
            refracted = -I * eta - N * (eta * cos_theta_i - cos_theta_t)
        return normalize(refracted), False
    else:
        # 全反射
        return ti.Vector([0.0, 0.0, 0.0]), True
```

同时使用Schlick近似计算菲涅尔效应，模拟光线在玻璃表面的反射与折射混合：

```python
@ti.func
def fresnel_schlick(cos_theta, eta):
    r0 = ((eta - 1.0) / (eta + 1.0)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cos_theta) ** 5
```

#### 3.2.4 抗锯齿

在每个像素内进行多次采样，并将结果平均：

```python
for s in range(samples):
    # 生成随机偏移
    rand_u = ti.math.fract(ti.sin(seed * 127.1 + 311.7) * 43758.5453123)
    rand_v = ti.math.fract(ti.sin(seed * 269.5 + 183.3) * 43758.5453123)
    # 计算采样点位置
    u = (i + offset_u - res_x/2) / res_y * 2.0
    v = (j + offset_v - res_y/2) / res_y * 2.0
    # 追踪光线并累加颜色
    final_color += trace_ray(ro, rd)
final_color /= samples  # 平均
```

### 3.3 交互控制

实现了以下UI控制功能：

1. **光源位置控制**：通过三个滑块动态调整点光源的X、Y、Z坐标
2. **最大弹射次数**：范围1-5，观察反射效果的层次变化
3. **MSAA开关**：启用或关闭抗锯齿
4. **采样数控制**：调整每像素采样次数（1-8）

## 四、运行录屏


## 五、总结

通过本次实验，我深入理解了光线追踪的核心原理，掌握了光线与几何体的求交算法，阴影、反射、折射等全局光照效果的实现方法，体会了GPU编程中迭代替代递归的思维方式，明白了交互式图形应用的开发流程。实验选做部分的玻璃材质和抗锯齿功能使画面效果有了质的提升，也让我体会到了图形学中真实感渲染的魅力。


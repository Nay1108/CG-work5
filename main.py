import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
msaa_enabled = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2
IOR_GLASS = 1.5

@ti.func
def normalize(v):
    return v / (v.norm() + 1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    cos_theta_i = ti.math.clamp(I.dot(N), -1.0, 1.0)
    sin_theta_i_sq = 1.0 - cos_theta_i * cos_theta_i
    sin_theta_t_sq = eta * eta * sin_theta_i_sq
    
    refracted_dir = ti.Vector([0.0, 0.0, 0.0])
    total_reflection = False
    
    if sin_theta_t_sq <= 1.0:
        cos_theta_t = ti.sqrt(1.0 - sin_theta_t_sq)
        if cos_theta_i < 0.0:
            refracted_dir = normalize(-I * eta + N * (eta * cos_theta_i - cos_theta_t))
        else:
            refracted_dir = normalize(-I * eta - N * (eta * cos_theta_i - cos_theta_t))
    else:
        total_reflection = True
    
    return refracted_dir, total_reflection

@ti.func
def fresnel_schlick(cos_theta, eta):
    r0 = (eta - 1.0) / (eta + 1.0)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cos_theta, 5.0)

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0.0:
        sqrt_delta = ti.sqrt(delta)
        t1 = (-b - sqrt_delta) / 2.0
        if t1 > 0.0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0.0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.5, 0.0, 0.0]), 1.0)
    if t > 0.0 and t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.7, 0.8, 0.9])
        hit_mat = MAT_GLASS

    t, n = intersect_sphere(ro, rd, ti.Vector([1.5, 0.0, 0.0]), 1.0)
    if t > 0.0 and t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    t, n = intersect_plane(ro, rd, -1.0)
    if t > 0.0 and t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        ix_int = ti.cast(ix, ti.i32)
        iz_int = ti.cast(iz, ti.i32)
        if (ix_int + iz_int) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def trace_reflect_ray(ro, rd, throughput):
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    final_color = ti.Vector([0.0, 0.0, 0.0])
    current_throughput = throughput
    current_ro = ro
    current_rd = rd

    for bounce in range(max_bounces[None]):
        t, N, obj_color, mat_id = scene_intersect(current_ro, current_rd)
        if t > 1e9:
            final_color += current_throughput * bg_color
            break
        p = current_ro + current_rd * t

        if mat_id == MAT_MIRROR:
            current_ro = p + N * 1e-4
            current_rd = normalize(reflect(current_rd, N))
            current_throughput *= 0.8 * obj_color
        elif mat_id == MAT_GLASS:
            current_ro = p + N * 1e-4
            current_rd = normalize(reflect(current_rd, N))
            current_throughput *= 0.9 * obj_color
        elif mat_id == MAT_DIFFUSE:
            L = normalize(light_pos - p)
            shadow_ray_orig = p + N * 1e-4
            shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
            dist_to_light = (light_pos - p).norm()
            in_shadow = 0.0
            if shadow_t < dist_to_light:
                in_shadow = 1.0
            ambient = 0.2 * obj_color
            direct_light = ambient
            if in_shadow == 0.0:
                diff = ti.max(0.0, N.dot(L))
                diffuse = 0.8 * diff * obj_color
                direct_light += diffuse
            final_color += current_throughput * direct_light
            break

    return final_color

@ti.func
def trace_ray(ro, rd):
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])
    current_ro = ro
    current_rd = rd

    for bounce in range(max_bounces[None]):
        t, N, obj_color, mat_id = scene_intersect(current_ro, current_rd)
        if t > 1e9:
            final_color += throughput * bg_color
            break
        p = current_ro + current_rd * t

        if mat_id == MAT_MIRROR:
            current_ro = p + N * 1e-4
            current_rd = normalize(reflect(current_rd, N))
            throughput *= 0.8 * obj_color

        elif mat_id == MAT_GLASS:
            # 提前初始化所有可能用到的变量
            entering = current_rd.dot(N) < 0.0
            eta = 0.0
            normal = ti.Vector([0.0, 0.0, 0.0])
            if entering:
                eta = 1.0 / IOR_GLASS
                normal = N
            else:
                eta = IOR_GLASS
                normal = -N

            refracted_dir, total_reflection = refract(current_rd, normal, eta)

            # 预定义变量，防止未使用时报错
            reflect_ro = ti.Vector([0.0, 0.0, 0.0])
            reflect_rd = ti.Vector([0.0, 0.0, 0.0])
            refract_ro = ti.Vector([0.0, 0.0, 0.0])
            refract_rd = ti.Vector([0.0, 0.0, 0.0])
            fresnel = 0.0

            if total_reflection:
                current_ro = p + N * 1e-4
                current_rd = normalize(reflect(current_rd, N))
                throughput *= 0.9 * obj_color
            else:
                cos_theta = ti.abs(current_rd.dot(N))
                fresnel = fresnel_schlick(cos_theta, IOR_GLASS)

                reflect_ro = p + N * 1e-4
                reflect_rd = normalize(reflect(current_rd, N))

                if entering:
                    refract_ro = p - N * 1e-4
                else:
                    refract_ro = p + N * 1e-4
                refract_rd = refracted_dir

                if bounce < max_bounces[None] - 1:
                    reflect_color = trace_reflect_ray(reflect_ro, reflect_rd,
                                                      throughput * fresnel * obj_color * 0.5)
                    final_color += reflect_color

                current_ro = refract_ro
                current_rd = refract_rd
                throughput *= (1.0 - fresnel) * 0.95 * obj_color

        elif mat_id == MAT_DIFFUSE:
            L = normalize(light_pos - p)
            shadow_ray_orig = p + N * 1e-4
            shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
            dist_to_light = (light_pos - p).norm()
            in_shadow = 0.0
            if shadow_t < dist_to_light:
                in_shadow = 1.0
            ambient = 0.2 * obj_color
            direct_light = ambient
            if in_shadow == 0.0:
                diff = ti.max(0.0, N.dot(L))
                diffuse = 0.8 * diff * obj_color
                direct_light += diffuse
            final_color += throughput * direct_light
            break

    return final_color

@ti.kernel
def render():
    for i, j in pixels:
        samples = samples_per_pixel[None] if msaa_enabled[None] == 1 else 1
        final_color = ti.Vector([0.0, 0.0, 0.0])

        for s in range(samples):
            seed = i * 1000 + j * 100 + s * 7 + 12345
            rand_u = ti.math.fract(ti.sin(float(seed) * 127.1 + 311.7) * 43758.5453123)
            rand_v = ti.math.fract(ti.sin(float(seed) * 269.5 + 183.3) * 43758.5453123)
            offset_u = (rand_u - 0.5) / float(res_y) * 2.0
            offset_v = (rand_v - 0.5) / float(res_y) * 2.0

            u = (float(i) - float(res_x) / 2.0 + offset_u * float(res_y)) / float(res_y) * 2.0
            v = (float(j) - float(res_y) / 2.0 + offset_v * float(res_y)) / float(res_y) * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))
            color = trace_ray(ro, rd)
            final_color += color

        final_color /= float(samples)
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing - Glass & MSAA", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3
    msaa_enabled[None] = 1
    samples_per_pixel[None] = 4

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.3):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)
            msaa_enabled[None] = 1 if gui.checkbox('Enable MSAA', msaa_enabled[None] == 1) else 0
            samples_per_pixel[None] = gui.slider_int('Samples', samples_per_pixel[None], 1, 8)

        window.show()

if __name__ == '__main__':
    main()
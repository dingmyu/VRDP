# -*- coding: utf-8 -*-
# Author: Mingyu Ding
# Time: 1/4/2021 12:44 PM
# Copyright 2019. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
import time
import json
import numpy as np
import torch
from LBFGS import FullBatchLBFGS


def get_2d_coor(x3d, y3d, z3d=0.2):
    cam_mat = np.array(((-207.8461456298828, 525.0000610351562, -120.00001525878906, 1200.0003662109375),
                        (123.93595886230469, 1.832598354667425e-05, -534.663330078125, 799.9999389648438),
                        (-0.866025447845459, -3.650024282819686e-08, -0.4999999701976776, 5.000000476837158),
                        (0, 0, 0, 1)))
    pos_3d = np.array([[x3d], [y3d], [z3d], [1.0]], dtype=np.float32)
    uv = cam_mat[:3].dot(pos_3d)
    pos_2d = uv[:-1] / uv[-1]
    return pos_2d


for process_index in range(int(sys.argv[1]), int(sys.argv[2])):
    object_dict = json.load(open(f'../data/object_dicts_with_physics/objects_{process_index:05d}.json'))
    output_dict = json.load(open(f'../data/object_simulated/sim_{process_index:05d}.json'))
    step_88 = output_dict['step_88']
    print(f'===============start processing {process_index}==================')

    device = 'cpu'

    n_balls = len(object_dict)
    steps = 210
    target_x = torch.zeros((128, n_balls, 2), dtype=torch.float32).to(device) + 1000

    shapes = []
    shape_dict = {
        'sphere': 0,
        'cube': 1,
        'cylinder': 2
    }

    for object_index, identity in enumerate(object_dict.keys()):
        locations = torch.tensor(object_dict[identity]['trajectory']).to(device)
        target_x[:locations.shape[0], object_index, :] = locations
        shapes.append(shape_dict[object_dict[identity]['shape']])

    target_x = target_x[-40:-19]
    for object_index, identity in enumerate(object_dict.keys()):
        if target_x[0][object_index][0] > 500:
            target_x[0][object_index] = torch.tensor(step_88['x'][object_index])

    shape = torch.tensor(shapes, dtype=torch.int8).to(device)
    angle0 = torch.tensor(step_88['angle'], dtype=torch.float32).to(device)
    angle0.requires_grad = True

    interval = 10
    dt = 1/350
    gravity = 9.806
    radius = 0.2
    inertia = 0.4 * 0.4 / 6

    frictional = torch.tensor(0.03).to(device)
    frictional.requires_grad = True
    linear_damping = torch.tensor(0.06).to(device)
    linear_damping.requires_grad = True
    v0 = torch.tensor(step_88['v'], dtype=torch.float32).to(device)
    v0.requires_grad = True

    restitution = torch.tensor(step_88['restitution'], dtype=torch.float32).to(device)
    restitution.requires_grad = True
    mass = torch.tensor(step_88['mass'], dtype=torch.float32).to(device)
    mass.requires_grad = True


    def norm(vector, degree=2, dim=0):
        return torch.norm(vector, degree, dim=dim)


    def normalized(vector):
        return vector / norm(vector)


    def collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions):
        imp = torch.tensor([0.0, 0.0]).to(device)
        x_inc_contrib = torch.tensor([0.0, 0.0]).to(device)
        if i != j:
            dist = (x[t, i] + dt * v[t, i]) - (x[t, j] + dt * v[t, j])
            dist_norm = norm(dist)
            rela_v = v[t, i] - v[t, j]
            if dist_norm < 2 * radius:
                dir = normalized(dist)
                projected_v = dir.dot(rela_v)

                if projected_v < 0:
                    if i < j:
                        repeat = False
                        for item in collisions:
                            if json.dumps(item).startswith(json.dumps([i, j])[:-1]):
                                repeat = True
                        if not repeat:
                            collisions.append([i, j, round(t / 10.0)])
                    imp = -(1 + restitution[i] * restitution[j]) * (mass[j] / (mass[i] + mass[j])) * projected_v * dir
                    toi = (dist_norm - 2 * radius) / min(
                        -1e-3, projected_v)
                    x_inc_contrib = min(toi - dt, 0) * imp
        x_inc[t + 1, i] += x_inc_contrib
        impulse[t + 1, i] += imp


    def sphere_collide_cube(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions):
        imp = torch.tensor([0.0, 0.0]).to(device)
        x_inc_contrib = torch.tensor([0.0, 0.0]).to(device)
        if i != j:
            rela_v = v[t, i] - v[t, j]
            pos_xy = x[t, i] - x[t, j]
            rotate_x = pos_xy.dot(torch.tensor([torch.cos(-angle[t, j]), -torch.sin(-angle[t, j])]))
            rotate_y = pos_xy.dot(torch.tensor([torch.sin(-angle[t, j]), torch.cos(-angle[t, j])]))
            moving_direction = torch.tensor([0.0, 0.0])
            dist_norm = 0.0
            collision = True

            if torch.abs(rotate_x) > 2 * radius:
                collision = False
            elif torch.abs(rotate_y) > 2 * radius:
                collision = False
            elif torch.abs(rotate_x) <= radius:
                if rotate_y > 0:
                    moving_direction = torch.tensor([0.0, 1.0])
                    dist_norm = rotate_y
                elif rotate_y < 0:
                    moving_direction = torch.tensor([0.0, -1.0])
                    dist_norm = - rotate_y
            elif torch.abs(rotate_y) <= radius:
                if rotate_x > 0:
                    moving_direction = torch.tensor([1.0, 0.0])
                    dist_norm = rotate_x
                elif rotate_x < 0:
                    moving_direction = torch.tensor([-1.0, 0.0])
                    dist_norm = - rotate_x
            elif (torch.abs(rotate_x) - radius) ** 2 + (torch.abs(rotate_y) - radius) ** 2 <= radius ** 2:
                if rotate_x > radius and rotate_y > radius:
                    moving_direction = normalized(torch.tensor([rotate_x - radius, rotate_y - radius]))
                    dist_norm = norm(torch.tensor([rotate_x - radius, rotate_y - radius])) + radius
                elif rotate_x < -radius and rotate_y > radius:
                    moving_direction = normalized(torch.tensor([rotate_x + radius, rotate_y - radius]))
                    dist_norm = norm(torch.tensor([rotate_x + radius, rotate_y - radius])) + radius
                elif rotate_x > radius and rotate_y < -radius:
                    moving_direction = normalized(torch.tensor([rotate_x - radius, rotate_y + radius]))
                    dist_norm = norm(torch.tensor([rotate_x - radius, rotate_y + radius])) + radius
                elif rotate_x < -radius and rotate_y < -radius:
                    moving_direction = normalized(torch.tensor([rotate_x + radius, rotate_y + radius]))
                    dist_norm = norm(torch.tensor([rotate_x + radius, rotate_y + radius])) + radius

            if collision:
                origin_dir = torch.tensor(
                    [moving_direction.dot(torch.tensor([torch.cos(angle[t, j]), -torch.sin(angle[t, j])])),
                     moving_direction.dot(torch.tensor([torch.sin(angle[t, j]), torch.cos(angle[t, j])]))]
                )
                projected_v = origin_dir.dot(rela_v)

                if projected_v < 0:
                    if i < j:
                        repeat = False
                        for item in collisions:
                            if json.dumps(item).startswith(json.dumps([i, j])[:-1]):
                                repeat = True
                        if not repeat:
                            collisions.append([i, j, round(t / 10.0)])
                    imp = -(1 + restitution[i] * restitution[j]) * (mass[j] / (mass[i] + mass[j])) * projected_v * origin_dir  # 冲量，速度变化量
                    toi = (dist_norm - 2 * radius) / min(
                        -1e-3, projected_v)
                    x_inc_contrib = min(toi - dt, 0) * imp

        x_inc[t + 1, i] += x_inc_contrib
        impulse[t + 1, i] += imp


    def cube_collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions):
        imp = torch.tensor([0.0, 0.0])
        x_inc_contrib = torch.tensor([0.0, 0.0])
        a_rotate = 0.0
        if i != j:
            rela_v = v[t, i] - v[t, j]
            pos_xy = x[t, j] - x[t, i]
            rotate_x = pos_xy.dot(torch.tensor([torch.cos(-angle[t, i]), -torch.sin(-angle[t, i])]))
            rotate_y = pos_xy.dot(torch.tensor([torch.sin(-angle[t, i]), torch.cos(-angle[t, i])]))

            moving_direction = torch.tensor([0.0, 0.0])
            collision_direction = torch.tensor([0.0, 0.0])
            dist_norm = 0.0
            r_rotate = 0.0
            rotate_dir = False
            collision = True

            if torch.abs(rotate_x) > 2 * radius:
                collision = False
            elif torch.abs(rotate_y) > 2 * radius:
                collision = False
            elif torch.abs(rotate_x) <= radius:
                if rotate_y > 0:
                    moving_direction = torch.tensor([0.0, -1.0])
                    collision_direction = normalized(torch.tensor([-rotate_x, -radius]))
                    dist_norm = rotate_y
                    if rotate_x > 0:
                        rotate_dir = 1
                elif rotate_y < 0:
                    moving_direction = torch.tensor([0.0, 1.0])
                    collision_direction = normalized(torch.tensor([-rotate_x, radius]))
                    dist_norm = - rotate_y
                    if rotate_x < 0:
                        rotate_dir = 1
                r_rotate = norm(torch.tensor([radius, rotate_x]))
            elif torch.abs(rotate_y) <= radius:
                if rotate_x > 0:
                    moving_direction = torch.tensor([-1.0, 0.0])
                    collision_direction = normalized(torch.tensor([-radius, -rotate_y]))
                    dist_norm = rotate_x
                    if rotate_y < 0:
                        rotate_dir = 1
                elif rotate_x < 0:
                    moving_direction = torch.tensor([1.0, 0.0])
                    collision_direction = normalized(torch.tensor([radius, -rotate_y]))
                    dist_norm = - rotate_x
                    if rotate_y > 0:
                        rotate_dir = 1
                r_rotate = norm(torch.tensor([radius, rotate_y]))
            elif (torch.abs(rotate_x) - radius) ** 2 + (torch.abs(rotate_y) - radius) ** 2 <= radius ** 2:
                if rotate_x > radius and rotate_y > radius:
                    moving_direction = - normalized(torch.tensor([rotate_x - radius, rotate_y - radius]))
                    collision_direction = normalized(torch.tensor([-1.0, -1.0]))
                    dist_norm = norm(torch.tensor([rotate_x - radius, rotate_y - radius])) + radius
                    if rotate_y > rotate_x:
                        rotate_dir = 1
                elif rotate_x < -radius and rotate_y > radius:
                    moving_direction = - normalized(torch.tensor([rotate_x + radius, rotate_y - radius]))
                    collision_direction = normalized(torch.tensor([1.0, -1.0]))
                    dist_norm = norm(torch.tensor([rotate_x + radius, rotate_y - radius])) + radius
                    if -rotate_x > rotate_y:
                        rotate_dir = 1
                elif rotate_x > radius and rotate_y < -radius:
                    moving_direction = - normalized(torch.tensor([rotate_x - radius, rotate_y + radius]))
                    collision_direction = normalized(torch.tensor([-1.0, 1.0]))
                    dist_norm = norm(torch.tensor([rotate_x - radius, rotate_y + radius])) + radius
                    if rotate_x > -rotate_y:
                        rotate_dir = 1
                elif rotate_x < -radius and rotate_y < -radius:
                    moving_direction = - normalized(torch.tensor([rotate_x + radius, rotate_y + radius]))
                    collision_direction = normalized(torch.tensor([1.0, 1.0]))
                    dist_norm = norm(torch.tensor([rotate_x + radius, rotate_y + radius])) + radius
                    if -rotate_y > -rotate_x:
                        rotate_dir = 1
                r_rotate = norm(torch.tensor([radius, radius]))

            if collision:
                origin_moving_dir = torch.tensor(
                    [moving_direction.dot(torch.tensor([torch.cos(angle[t, i]), -torch.sin(angle[t, i])])),
                     moving_direction.dot(torch.tensor([torch.sin(angle[t, i]), torch.cos(angle[t, i])]))]
                )
                origin_collision_dir = torch.tensor(
                    [collision_direction.dot(torch.tensor([torch.cos(angle[t, i]), -torch.sin(angle[t, i])])),
                     collision_direction.dot(torch.tensor([torch.sin(angle[t, i]), torch.cos(angle[t, i])]))]
                )
                projected_v = origin_moving_dir.dot(rela_v)

                if projected_v < 0:
                    if i < j:
                        repeat = False
                        for item in collisions:
                            if json.dumps(item).startswith(json.dumps([i, j])[:-1]):
                                repeat = True
                        if not repeat:
                            collisions.append([i, j, round(t / 10.0)])
                    imp = -(1 + restitution[i] * restitution[j]) * (mass[j] / (mass[i] + mass[j])) * projected_v * origin_moving_dir
                    toi = (dist_norm - 2 * radius) / min(
                        -1e-3, projected_v)
                    x_inc_contrib = min(toi - dt, 0) * imp

                    f_rotate = (origin_moving_dir - origin_collision_dir.dot(origin_moving_dir) * origin_collision_dir).dot(-projected_v * origin_moving_dir)
                    a_rotate = f_rotate * r_rotate / inertia
                    if rotate_dir:
                        a_rotate = -a_rotate

        x_inc[t + 1, i] += x_inc_contrib
        impulse[t + 1, i] += imp
        angle_impulse[t + 1, i] += a_rotate


    def collide(shape, x, v, x_inc, impulse, t, angle, angle_impulse, collisions):
        for i in range(n_balls):
            for j in range(i):
                if shape[i] != 1 and shape[j] != 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)
                elif shape[i] != 1 and shape[j] == 1:
                    sphere_collide_cube(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)
                elif shape[i] == 1 and shape[j] != 1:
                    cube_collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)
                elif shape[i] == 1 and shape[j] == 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)

        for i in range(n_balls):
            for j in range(i + 1, n_balls):
                if shape[i] != 1 and shape[j] != 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)
                elif shape[i] != 1 and shape[j] == 1:
                    sphere_collide_cube(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)
                elif shape[i] == 1 and shape[j] != 1:
                    cube_collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)
                elif shape[i] == 1 and shape[j] == 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse, collisions)


    def friction(shape, x, v, x_inc, impulse, v_old, t, i):
        if shape[i] == 0:
            if v_old[0] > 0.0:
                v[t, i][0] = max(0, v_old[0] - linear_damping * dt * v_old[0] * norm(v_old))
            elif v_old[0] < 0.0:
                v[t, i][0] = min(0, v_old[0] - linear_damping * dt * v_old[0] * norm(v_old))
            if v_old[1] > 0.0:
                v[t, i][1] = max(0, v_old[1] - linear_damping * dt * v_old[1] * norm(v_old))
            elif v_old[1] < 0.0:
                v[t, i][1] = min(0, v_old[1] - linear_damping * dt * v_old[1] * norm(v_old))
        else:
            if v_old[0] > 0.0:
                v[t, i][0] = max(0, v_old[0] - gravity * frictional * dt * normalized(v_old)[0] - linear_damping * dt * v_old[0] * norm(v_old))
            elif v_old[0] < 0.0:
                v[t, i][0] = min(0, v_old[0] - gravity * frictional * dt * normalized(v_old)[0] - linear_damping * dt * v_old[0] * norm(v_old))
            if v_old[1] > 0.0:
                v[t, i][1] = max(0, v_old[1] - gravity * frictional * dt * normalized(v_old)[1] - linear_damping * dt * v_old[1] * norm(v_old))
            elif v_old[1] < 0.0:
                v[t, i][1] = min(0, v_old[1] - gravity * frictional * dt * normalized(v_old)[1] - linear_damping * dt * v_old[1] * norm(v_old))


    def advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse):
        for i in range(n_balls):
            v_old = v[t - 1, i] + impulse[t, i]
            friction(shape, x, v, x_inc, impulse, v_old, t, i)
            x[t, i] = x[t - 1, i] + dt * (v[t, i] + v_old)/2 + x_inc[t, i]
            delta_angle[t, i] = delta_angle[t - 1, i] + angle_impulse[t, i]
            if delta_angle[t, i] > 0.0:
                delta_angle[t, i] = max(0, delta_angle[t, i] - dt * gravity / 2)
            elif delta_angle[t, i] < 0.0:
                delta_angle[t, i] = min(0, delta_angle[t, i] + dt * gravity / 2)
            angle[t, i] = angle[t - 1, i] + dt * delta_angle[t, i]

    def init():
        x = torch.zeros((steps, n_balls, 2), dtype=torch.float32).to(device)
        v = torch.zeros((steps, n_balls, 2), dtype=torch.float32).to(device)
        x_inc = torch.zeros((steps, n_balls, 2), dtype=torch.float32).to(device)
        impulse = torch.zeros((steps, n_balls, 2), dtype=torch.float32).to(device)
        angle = torch.zeros((steps, n_balls), dtype=torch.float32).to(device)
        delta_angle = torch.zeros((steps, n_balls), dtype=torch.float32).to(device)
        angle_impulse = torch.zeros((steps, n_balls), dtype=torch.float32).to(device)

        x[0, :] = target_x[0]
        v[0, :] = v0
        angle[0, :] = angle0
        return x, v, x_inc, impulse, angle, delta_angle, angle_impulse


    def closure():
        optimizer.zero_grad()
        x, v, x_inc, impulse, angle, delta_angle, angle_impulse = init()
        loss = 0
        collisions = []
        for t in range(1, 210):
            collide(shape, x, v, x_inc, impulse, t - 1, angle, angle_impulse, collisions)
            advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)

            if t % interval == 0:
                loss += (((x[t, :] - target_x[int(t/interval), :]) * (target_x[int(t/interval), :] < 100)) ** 2).mean()
        return loss


    def init_inference():
        x = torch.zeros((210, n_balls, 2), dtype=torch.float32).to(device)
        v = torch.zeros((210, n_balls, 2), dtype=torch.float32).to(device)
        x_inc = torch.zeros((210, n_balls, 2), dtype=torch.float32).to(device)
        impulse = torch.zeros((210, n_balls, 2), dtype=torch.float32).to(device)
        angle = torch.zeros((210, n_balls), dtype=torch.float32).to(device)
        delta_angle = torch.zeros((210, n_balls), dtype=torch.float32).to(device)
        angle_impulse = torch.zeros((210, n_balls), dtype=torch.float32).to(device)

        x[0, :] = target_x[0]
        v[0, :] = v0
        angle[0, :] = angle0
        return x, v, x_inc, impulse, angle, delta_angle, angle_impulse

# if __name__ == '__main__':
    optimizer = FullBatchLBFGS([v0, mass, restitution])
    start = time.time()
    loss = closure()
    loss.backward()

    for i in range(15):
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
        print(loss, lr, v0, mass, restitution)
        if loss < 0.0002 or lr == 0:
            break

    time_cost = time.time() - start
    print(f'----- learned, cost {time_cost}s')

    collisions = []
    x, v, x_inc, impulse, angle, delta_angle, angle_impulse = init_inference()
    for t in range(1, 210):
        collide(shape, x, v, x_inc, impulse, t - 1, angle, angle_impulse, collisions)  # 计算碰撞
        advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)  # 更新速度和位置

    # ==================================================================================


    shapes = []
    shape_dict = {
        'sphere': 0,
        'cube': 1,
        'cylinder': 2
    }
    reverse_shape_dict = {
        0: 'sphere',
        1: 'cube',
        2: 'cylinder'
    }
    colors = []
    materials = []
    for object_index, identity in enumerate(object_dict.keys()):
        shapes.append(shape_dict[object_dict[identity]['shape']])
        colors.append(object_dict[identity]['color'])
        materials.append(object_dict[identity]['material'])

    gt_objects = list(object_dict.keys())
    old_collisions = output_dict['predictions'][0]['collisions'].copy()
    uniq_collisions = []
    for item in old_collisions:
        if item['frame'] > 88:
            output_dict['predictions'][0]['collisions'].remove(item)
            print('remove collision', item['frame'])
        else:
            uniq_collisions.append([gt_objects.index(item['objects'][0]['color'] + item['objects'][0]['material'] + item['objects'][0]['shape']),
                                   gt_objects.index(item['objects'][1]['color'] + item['objects'][1]['material'] + item['objects'][1]['shape']),
                                   item['frame']])

    for collision_index, item in enumerate(collisions):
        i, j, frame = item
        repeat = False
        for colli_item in uniq_collisions:
            if json.dumps(colli_item).startswith(json.dumps([i, j])[:-1]):
                repeat = True
        if not repeat:
            output_dict['predictions'][0]['collisions'].append({
                'frame': 88 + frame,
                'objects': [{
                    'color': colors[i],
                    'material': materials[i],
                    'shape': reverse_shape_dict[shapes[i]],
                }, {
                    'color': colors[j],
                    'material': materials[j],
                    'shape': reverse_shape_dict[shapes[j]],
                }]
            })
            print('add collision', 88 + frame)

    output_dict['predictions'][0]['trajectory'] = output_dict['predictions'][0]['trajectory'][:18]
    print('keep trajectory from 0 to', output_dict['predictions'][0]['trajectory'][-1]['frame_index'])
    for frame_index, locations in enumerate(x):
        if frame_index % 50 == 20:
            frame_info = {'frame_index': 88 + frame_index // 10,
                          'objects': []}
            for object_index, location in enumerate(locations):
                xy = get_2d_coor(location[0].cpu().item(), location[1].cpu().item())
                xy1 = get_2d_coor(location[0].cpu().item() + radius * 0.7071, location[1].cpu().item(), z3d=radius * (1 - 0.7071))
                xy2 = get_2d_coor(location[0].cpu().item() - radius * 0.7071, location[1].cpu().item(), z3d=radius * (1 + 0.7071))
                xy3 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item() + radius)
                xy4 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item() - radius)
                xy5 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item(), z3d=0)
                xy6 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item(), z3d=2 * radius)
                if (-10 < xy[0] < 490 and -10 < xy[1] < 330) \
                        or (0 < xy1[0] < 480 and 0 < xy1[1] < 320) \
                        or (0 < xy2[0] < 480 and 0 < xy2[1] < 320) \
                        or (0 < xy3[0] < 480 and 0 < xy3[1] < 320) \
                        or (0 < xy4[0] < 480 and 0 < xy3[1] < 320) \
                        or (0 < xy5[0] < 480 and 0 < xy3[1] < 320) \
                        or (0 < xy6[0] < 480 and 0 < xy4[1] < 320):
                    frame_info['objects'].append({
                        'x': float(xy[1]) / 3.2,
                        'y': float(xy[0]) / 3.2,
                        'color': colors[object_index],
                        'material': materials[object_index],
                        'shape': reverse_shape_dict[shapes[object_index]],
                    })

            output_dict['predictions'][0]['trajectory'].append(frame_info)
            print('add trajectory', frame_info['frame_index'])

    n_balls = len(object_dict)
    steps = 200
    target_x = torch.zeros((128, n_balls, 2), dtype=torch.float32).to(device) + 1000

    shapes = []
    shape_dict = {
        'sphere': 0,
        'cube': 1,
        'cylinder': 2
    }

    for object_index, identity in enumerate(object_dict.keys()):
        locations = torch.tensor(object_dict[identity]['trajectory']).to(device)
        target_x[:locations.shape[0], object_index, :] = locations
        shapes.append(shape_dict[object_dict[identity]['shape']])

    target_x = target_x[-20:]
    for object_index, identity in enumerate(object_dict.keys()):
        if target_x[0][object_index][0] > 500:
            target_x[0][object_index] = torch.tensor(x[-1].detach()[object_index])

    shape = torch.tensor(shapes, dtype=torch.int8).to(device)
    angle0 = angle[-1].detach()
    angle0.requires_grad = True

    interval = 10
    dt = 1/350
    gravity = 9.806
    radius = 0.2
    inertia = 0.4 * 0.4 / 6

    frictional = torch.tensor(0.03).to(device)
    frictional.requires_grad = True
    linear_damping = torch.tensor(0.06).to(device)
    linear_damping.requires_grad = True
    v0 = torch.tensor(v[-1].detach(), dtype=torch.float32).to(device)
    v0.requires_grad = True

    restitution = torch.tensor(restitution.detach(), dtype=torch.float32).to(device)
    restitution.requires_grad = True
    mass = torch.tensor(mass.detach(), dtype=torch.float32).to(device)
    mass.requires_grad = True


    def closure_108():
        optimizer.zero_grad()
        x, v, x_inc, impulse, angle, delta_angle, angle_impulse = init()
        loss = 0
        collisions = []
        for t in range(1, 200):
            collide(shape, x, v, x_inc, impulse, t - 1, angle, angle_impulse, collisions)
            advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)

            if t % interval == 0:
                loss += (((x[t, :] - target_x[int(t/interval), :]) * (target_x[int(t/interval), :] < 100)) ** 2).mean()
        return loss


    def init_inference_108():
        x = torch.zeros((780, n_balls, 2), dtype=torch.float32).to(device)
        v = torch.zeros((780, n_balls, 2), dtype=torch.float32).to(device)
        x_inc = torch.zeros((780, n_balls, 2), dtype=torch.float32).to(device)
        impulse = torch.zeros((780, n_balls, 2), dtype=torch.float32).to(device)
        angle = torch.zeros((780, n_balls), dtype=torch.float32).to(device)
        delta_angle = torch.zeros((780, n_balls), dtype=torch.float32).to(device)
        angle_impulse = torch.zeros((780, n_balls), dtype=torch.float32).to(device)

        x[0, :] = target_x[0]
        v[0, :] = v0
        angle[0, :] = angle0
        return x, v, x_inc, impulse, angle, delta_angle, angle_impulse


    optimizer = FullBatchLBFGS([v0, mass, restitution])
    start = time.time()
    loss = closure_108()
    loss.backward()

    for i in range(15):
        options = {'closure': closure_108, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
        print(loss, lr, v0, mass, restitution)
        if loss < 0.0002 or lr == 0:
            break

    time_cost = time.time() - start
    print(f'----- learned, cost {time_cost}s')

    collisions = []
    x, v, x_inc, impulse, angle, delta_angle, angle_impulse = init_inference_108()
    for t in range(1, 780):
        collide(shape, x, v, x_inc, impulse, t - 1, angle, angle_impulse, collisions)
        advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)

    # ==================================================================================


    shapes = []
    shape_dict = {
        'sphere': 0,
        'cube': 1,
        'cylinder': 2
    }
    reverse_shape_dict = {
        0: 'sphere',
        1: 'cube',
        2: 'cylinder'
    }
    colors = []
    materials = []
    for object_index, identity in enumerate(object_dict.keys()):
        shapes.append(shape_dict[object_dict[identity]['shape']])
        colors.append(object_dict[identity]['color'])
        materials.append(object_dict[identity]['material'])

    gt_objects = list(object_dict.keys())
    old_collisions = output_dict['predictions'][0]['collisions'].copy()
    uniq_collisions = []
    for item in old_collisions:
        if item['frame'] > 108:
            output_dict['predictions'][0]['collisions'].remove(item)
            print('remove collision', item['frame'])
        else:
            uniq_collisions.append([gt_objects.index(item['objects'][0]['color'] + item['objects'][0]['material'] + item['objects'][0]['shape']),
                                    gt_objects.index(item['objects'][1]['color'] + item['objects'][1]['material'] + item['objects'][1]['shape']),
                                    item['frame']])

    for collision_index, item in enumerate(collisions):
        i, j, frame = item
        repeat = False
        for colli_item in uniq_collisions:
            if json.dumps(colli_item).startswith(json.dumps([i, j])[:-1]):
                repeat = True
        if not repeat:
            output_dict['predictions'][0]['collisions'].append({
                'frame': 108 + frame,
                'objects': [{
                    'color': colors[i],
                    'material': materials[i],
                    'shape': reverse_shape_dict[shapes[i]],
                }, {
                    'color': colors[j],
                    'material': materials[j],
                    'shape': reverse_shape_dict[shapes[j]],
                }]
            })
            print('add collision', 108 + frame)

    output_dict['predictions'][0]['trajectory'] = output_dict['predictions'][0]['trajectory'][:22]
    print('keep trajectory from 0 to', output_dict['predictions'][0]['trajectory'][-1]['frame_index'])
    for frame_index, locations in enumerate(x):
        if frame_index % 50 == 20:
            frame_info = {'frame_index': 108 + frame_index // 10,
                          'objects': []}
            for object_index, location in enumerate(locations):
                xy = get_2d_coor(location[0].cpu().item(), location[1].cpu().item())
                xy1 = get_2d_coor(location[0].cpu().item() + radius * 0.7071, location[1].cpu().item(), z3d=radius * (1 - 0.7071))
                xy2 = get_2d_coor(location[0].cpu().item() - radius * 0.7071, location[1].cpu().item(), z3d=radius * (1 + 0.7071))
                xy3 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item() + radius)
                xy4 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item() - radius)
                xy5 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item(), z3d=0)
                xy6 = get_2d_coor(location[0].cpu().item(), location[1].cpu().item(), z3d=2 * radius)
                if (-10 < xy[0] < 490 and -10 < xy[1] < 330) \
                        or (0 < xy1[0] < 480 and 0 < xy1[1] < 320) \
                        or (0 < xy2[0] < 480 and 0 < xy2[1] < 320) \
                        or (0 < xy3[0] < 480 and 0 < xy3[1] < 320) \
                        or (0 < xy4[0] < 480 and 0 < xy3[1] < 320) \
                        or (0 < xy5[0] < 480 and 0 < xy3[1] < 320) \
                        or (0 < xy6[0] < 480 and 0 < xy4[1] < 320):
                    frame_info['objects'].append({
                        'x': float(xy[1]) / 3.2,
                        'y': float(xy[0]) / 3.2,
                        'color': colors[object_index],
                        'material': materials[object_index],
                        'shape': reverse_shape_dict[shapes[object_index]],
                    })

            output_dict['predictions'][0]['trajectory'].append(frame_info)
            print('add trajectory', frame_info['frame_index'])

    json.dump(output_dict, open(f'../data/object_updated_results/sim_{process_index:05d}.json', 'w'))


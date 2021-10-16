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
import torch
import json
import time
from LBFGS import FullBatchLBFGS
import sys


for process_index in range(int(sys.argv[1]), int(sys.argv[2])):
    object_dict = json.load(open(f'../data/object_dicts/objects_{process_index:05d}.json'))
    print(f'===============start processing {process_index}==================')

    device = 'cpu'

    n_balls = len(object_dict)
    steps = 1280
    target_x = torch.zeros((128, n_balls, 2), dtype=torch.float32).to(device) + 1000

    shapes = []
    shape_dict = {
        'sphere': 0,
        'cube': 1,
        'cylinder': 2
    }
    angles = []
    for object_index, identity in enumerate(object_dict.keys()):
        locations = torch.tensor(object_dict[identity]['start_locations']).to(device)
        target_x[:locations.shape[0], object_index, :] = locations
        shapes.append(shape_dict[object_dict[identity]['shape']])
        angles.append(object_dict[identity]['initial_orientation'])

    shape = torch.tensor(shapes, dtype=torch.int8).to(device)
    angle0 = torch.tensor(angles, dtype=torch.float32).to(device)
    angle0.requires_grad = True

    interval = 10
    dt = 1/350
    gravity = 9.806
    radius = torch.tensor(0.2)
    radius.requires_grad = True
    inertia = 0.4 * 0.4 / 6

    frictional = torch.tensor(0.03).to(device)
    frictional.requires_grad = True
    linear_damping = torch.tensor(0.06).to(device)
    linear_damping.requires_grad = True
    v0 = torch.zeros((n_balls, 2), dtype=torch.float32).to(device)
    v0.requires_grad = True

    restitution = torch.zeros((n_balls), dtype=torch.float32).to(device) + 0.6
    restitution.requires_grad = True
    mass = torch.zeros((n_balls), dtype=torch.float32).to(device) + 1
    mass.requires_grad = True


    def norm(vector, degree=2, dim=0):
        return torch.norm(vector, degree, dim=dim)


    def normalized(vector):
        return vector / norm(vector)


    def collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse):
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
                    imp = -(1 + restitution[i] * restitution[j]) * (mass[j] / (mass[i] + mass[j])) * projected_v * dir
                    toi = (dist_norm - 2 * radius) / min(
                        -1e-3, projected_v)
                    x_inc_contrib = min(toi - dt, 0) * imp
        x_inc[t + 1, i] += x_inc_contrib
        impulse[t + 1, i] += imp


    def sphere_collide_cube(x, v, x_inc, impulse, t, i, j, angle, angle_impulse):
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
                    imp = -(1 + restitution[i] * restitution[j]) * (mass[j] / (mass[i] + mass[j])) * projected_v * origin_dir
                    toi = (dist_norm - 2 * radius) / min(
                        -1e-3, projected_v)
                    x_inc_contrib = min(toi - dt, 0) * imp

        x_inc[t + 1, i] += x_inc_contrib
        impulse[t + 1, i] += imp


    def cube_collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse):
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
                    imp = -(1 + restitution[i] * restitution[j]) * (mass[j] / (mass[i] + mass[j])) * projected_v * origin_moving_dir  # 冲量，速度变化量
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


    def collide(shape, x, v, x_inc, impulse, t, angle, angle_impulse):
        for i in range(n_balls):
            for j in range(i):
                if shape[i] != 1 and shape[j] != 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)
                elif shape[i] != 1 and shape[j] == 1:
                    sphere_collide_cube(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)
                elif shape[i] == 1 and shape[j] != 1:
                    cube_collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)
                elif shape[i] == 1 and shape[j] == 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)

        for i in range(n_balls):
            for j in range(i + 1, n_balls):
                if shape[i] != 1 and shape[j] != 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)
                elif shape[i] != 1 and shape[j] == 1:
                    sphere_collide_cube(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)
                elif shape[i] == 1 and shape[j] != 1:
                    cube_collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)
                elif shape[i] == 1 and shape[j] == 1:
                    collide_sphere(x, v, x_inc, impulse, t, i, j, angle, angle_impulse)


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
        for t in range(1, 1280):
            advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)

            if t % interval == 0:
                loss += (((x[t, :] - target_x[int(t/interval), :]) * (target_x[int(t/interval), :] < 100)) ** 2).mean()
        return loss


    def closure_full_1():
        optimizer.zero_grad()
        x, v, x_inc, impulse, angle, delta_angle, angle_impulse = init()
        loss = 0
        for t in range(1, 400):
            collide(shape, x, v, x_inc, impulse, t - 1, angle, angle_impulse)
            advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)

            if t % interval == 0:
                loss += (((x[t, :] - target_x[int(t/interval), :]) * (target_x[int(t/interval), :] < 100)) ** 2).mean()
        return loss

    def closure_full_2():
        optimizer.zero_grad()
        x, v, x_inc, impulse, angle, delta_angle, angle_impulse = init()
        loss = 0
        for t in range(1, 800):
            collide(shape, x, v, x_inc, impulse, t - 1, angle, angle_impulse)
            advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)

            if t % interval == 0:
                loss += (((x[t, :] - target_x[int(t/interval), :]) * (target_x[int(t/interval), :] < 100)) ** 2).mean()
        return loss

    def closure_full():
        optimizer.zero_grad()
        x, v, x_inc, impulse, angle, delta_angle, angle_impulse = init()
        loss = 0
        for t in range(1, 1280):
            collide(shape, x, v, x_inc, impulse, t - 1, angle, angle_impulse)
            advance(shape, x, v, x_inc, impulse, t, angle, delta_angle, angle_impulse)

            if t % interval == 0:
                loss += (((x[t, :] - target_x[int(t/interval), :]) * (target_x[int(t/interval), :] < 100)) ** 2).mean()
        return loss


# if __name__ == '__main__':
    # ------------- learn initial velocity --------
    optimizer = FullBatchLBFGS([v0])
    start = time.time()
    loss = closure()
    loss.backward()

    loss_velocity = 100
    for i in range(15):
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
        print(loss, lr, v0)
        if loss < loss_velocity:
            loss_velocity = loss.item()
        if loss < 0.0005 or lr == 0:
            break

    time_cost = time.time() - start
    print(f'----- velocity is learned, cost {time_cost}s')

    # ------------- learn mass and restitution --------
    for object_index, identity in enumerate(object_dict.keys()):
        locations = torch.tensor(object_dict[identity]['trajectory']).to(device)
        target_x[:locations.shape[0], object_index, :] = locations

    optimizer = FullBatchLBFGS([v0, mass, restitution])
    start = time.time()
    loss = closure_full_1()
    loss.backward()

    for i in range(20):
        options = {'closure': closure_full_1, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
        print(loss, lr, v0, mass, restitution)
        if loss < 0.0002 or lr == 0:
            break
    print(f'----- step1 is learned, loss {loss}')

    optimizer = FullBatchLBFGS([mass, restitution])
    loss = closure_full_2()
    loss.backward()
    for i in range(20):
        options = {'closure': closure_full_2, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
        print(loss, lr, mass, restitution)
        if loss < 0.001 or lr == 0:
            break
    print(f'----- step2 is learned, loss {loss}')

    optimizer = FullBatchLBFGS([mass, restitution])
    loss = closure_full()
    loss.backward()

    loss_full = 100
    for i in range(20):
        options = {'closure': closure_full, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
        if loss < loss_full:
            loss_full = loss.item()
        print(loss, lr, mass, restitution)
        if loss < 0.01 or lr == 0:
            break

    time_cost = time.time() - start
    print(f'----- mass and restitution are learned, loss {loss_full}, cost {time_cost}s')

    # ------------- update information --------

    gt_objects = list(object_dict.keys())

    for object_index, velocity in enumerate(v0):
        object_dict[gt_objects[object_index]]['initial_velocity'] = velocity.cpu().tolist()
    for object_index, object_restitution in enumerate(restitution):
        object_dict[gt_objects[object_index]]['restitution'] = object_restitution.cpu().item()
    for object_index, object_mass in enumerate(mass):
        object_dict[gt_objects[object_index]]['mass'] = object_mass.cpu().item()

        object_dict[gt_objects[object_index]]['loss_velocity'] = loss_velocity
        object_dict[gt_objects[object_index]]['loss_full'] = loss_full

    json.dump(object_dict, open(f'../data/object_dicts_with_physics/objects_{process_index:05d}.json', 'w'))

#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import intention
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import sys
import time
import math
# from scipy import integrate
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
import numpy
from numpy import random
random.seed(13)
import threading

import torch
from deploy import traj_pred
from model import highwayNet
from deploy import call_model

def getlinearline(p1, p2):
    return lambda x: (x-p1[0])/(p2[0]-p1[0])*(p2[1]-p1[1])+p1[1]

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=0,
        type=int,
        help='Number of vehicles (default: 100)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=2000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        # traffic_manager.global_percentage_speed_difference(-200)
        # traffic_manager.set_hybrid_physics_mode(True)
        # traffic_manager.set_hybrid_physics_radius(70)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(False)
            # traffic_manager.set_hybrid_physics_radius(70.0)

        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 2]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()

        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            pass#random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []

        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = blueprints[0]
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            #spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
             .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)






        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))


        traffic_manager.global_percentage_speed_difference(50)
        R=50
        D=1
        n=0
        num = 1
        num = int(num)
        x = numpy.zeros((15, 3, num))
        matrix_all = numpy.zeros((15, 4, 4))


        # print(vehicles_list)





        for actor in world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                player = actor
                break

        # carla.Rotation.__init__(player,pitch=0.0, yaw=0.0, roll=0.0)
        # print(player.get_transform().rotation)
        # ego_len = 4.8
        # ego_wid = 1.8
        # traffic_len = 4.8
        # traffic_wid = 1.8
        # sigma = 1
        # spectator = world.get_spectator()
        # camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

        # trajectory prediction model initialization
        # args = {}
        # args['use_cuda'] = True
        # args['encoder_size'] = 64
        # args['decoder_size'] = 128
        # args['in_length'] = 16
        # args['out_length'] = 25
        # args['grid_size'] = (13, 3)
        # args['soc_conv_depth'] = 64
        # args['conv_3x1_depth'] = 16
        # args['dyn_embedding_size'] = 32
        # args['input_embedding_size'] = 32
        # args['num_lat_classes'] = 3
        # args['num_lon_classes'] = 2
        # args['use_maneuvers'] = False
        # args['train_flag'] = False
        #
        # args = initialize_args()
        # model = highwayNet(args)
        # model.load_state_dict(torch.load('trained_models/cslstm_m_0.pt'))
        # model = highwayNet(args)
        # model = model()

        while True:



            if not args.asynch and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()


            # actor_list = world.get_actors(vehicles_list)
            # for a in range(len(vehicles_list)):
            #     actor = actor_list.find(vehicles_list[a])
            #     #print('id: ', vehicles_list[a], actor.get_location().x, actor.get_velocity().x)
            #     print()
            #
            #     with open('C:/carla/WindowsNoEditor/PythonAPI/examples/src/data/location.txt', 'a+') as f:
            #         print ('id: ', vehicles_list[a], actor.get_location(), actor.get_velocity(),file=f)
            # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            # camera = world.spawn_actor(camera_bp, camera_transform, attach_to=player)
            # spectator.set_transform(camera.get_transform())
            matrix = player.get_transform().get_matrix() #ego to global
            # matrix = [[9.99979339e-01, -6.42989214e-03, 9.53674252e-07, 3.66073886e+02],
            #           [6.42989220e-03, 9.99979338e-01, -3.04334952e-05, -3.12205834e+01],
            #           [-7.57970461e-07, 3.04389985e-05, 9.99999999e-01, -3.23974599e-03],
            #           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
            n = n + 1


            actor_list = world.get_actors(vehicles_list)
            # ego = actor_list.find(vehicles_list[0])
            player_info = player.get_transform()
            matrix = numpy.linalg.inv(matrix) # global to ego
            inv_matrix = player.get_transform().get_matrix()  # ego to global
            inv_matrix = numpy.array(inv_matrix)
            # print(inverse_matrix)
            print('player', player_info.location)

            # orientation = [player.get_transform().rotation.roll, player.get_transform().rotation.pitch, player.get_transform().rotation.yaw-player.get_transform().rotation.yaw]



            player_location = [[player_info.location.x], [player_info.location.y], [player_info.location.z],
                               [1]] #global frame
            # player_velocity = [[player.get_velocity().x], [player.get_velocity().y], [player.get_velocity().z], [0]]
            player_final_location = numpy.dot(matrix,player_location)
           #ego frame, should be zero

            # player_final_velocity = numpy.dot(matrix,player_velocity)

            # print('location', player_final_location, 'velocity', player_final_velocity)
            # print("location:", player_final_location[0][0],  player_final_location[1][0])
            #
            # spectator_location = [[player_final_location[0][0]  ], [player_final_location[1][0]+20],
            #                       [player_final_location[2][0]+20], [1]]
            #
            # spectator_real_location = numpy.dot(matrix, spectator_location)
            # spectator_real_location_carla = carla.Location(spectator_real_location[0][0], spectator_real_location[1][0],
            #                                                spectator_real_location[2][0])
            # spectator = world.get_spectator()
            # spectator.set_transform(carla.Transform(spectator_real_location_carla, carla.Rotation(pitch=-90)))
            # spectator = world.get_spectator()
            # transform = player.get_transform()
            # spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))
            # for n in range(6):
            #     future_location= [[player_final_location[0][0]+3*n], [player_final_location[1][0]], [player_final_location[2][0]],[1]]
            #
            #
            #     future_real_location = numpy.dot(matrix,future_location)
            #     future_real_location_carla = carla.Location(future_real_location[0][0], future_real_location[1][0], future_real_location[2][0])
            #     world.debug.draw_point(future_real_location_carla, size=0.1, color=carla.Color(r=0, g=255, b=0), life_time=5)
            #
            # print('player', 'location:', player_final_location[0][0], player_final_location[1][0], player_final_location[2][0],  'velocity:', player_final_velocity[0][0], player_final_velocity[1][0], player_final_velocity[2][0])
            #
            # world.debug.draw_point(player_info.location, size=0.1, color=carla.Color(r=255, g=0, b=0), life_time=5)
            # # world.debug.draw_string(player_info.location, '^', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0)
            # vehicles = []
            # # print(manual_control_joystick_copy.game_loop().location)

            if n <= 75:
                if n % 5 == 0:
                    t = n/5-1
                    t = int(t)
                    for a in range(len(vehicles_list)):
                        actor = actor_list.find(vehicles_list[a])

                    # print('id: ', vehicles_list[a], actor.get_location(), actor.get_velocity())
                #     # with open('C:/carla/WindowsNoEditor/PythonAPI/examples/src/data/location.txt', 'a+') as f:
                #     #     print('id: ', vehicles_list[a], actor.get_location(), actor.get_velocity(), file=f)
                #     if ((player.get_location().x-actor.get_location().x)**2+(player.get_location().y-actor.get_location().y)**2) <= (R**2):
                #         # actor.apply_control(carla.VehicleControl(throttle=0, brake=1))
                #         print('actorOrientation: ', actor.get_transform().rotation.yaw-player.get_transform().rotation.yaw)
                #     #     print('id: ', vehicles_list[a], actor.get_location(), actor.get_velocity(), len(vehicles))
                #         vehicles.append(vehicles_list[a])
                        actor_location = [[actor.get_location().x], [actor.get_location().y], [actor.get_location().z], [1]]
                        # actor_velocity =[[actor.get_velocity().x], [actor.get_velocity().y], [actor.get_velocity().z], [0]]
                        # print("velocity1", actor.get_velocity())
                        # print(actor.get_transform().rotation)
                        actor_final_location = numpy.dot(matrix, actor_location)



                        matrix_all[t,:,:] = inv_matrix

                        # actor_final_velocity = numpy.dot(matrix, actor_velocity)
                        # print("velocity", actor_final_velocity[0][0], actor_final_velocity[1][0])
                        x[t, 0, a] = actor_final_location[0][0]
                        x[t, 1, a] = actor_final_location[1][0]
                        x[t, 2, a] = actor_final_location[2][0]
                        # x[t, 2, a] = actor_final_velocity[0][0]
                        # x[t, 3, a] = actor_final_velocity[1][0]
                    x[t, 0, num-1] = player_final_location[0][0]
                    x[t, 1, num-1] = player_final_location[1][0]
                    x[t, 2, num-1] = player_final_location[2][0]
                    # x[t, 2, 50] = player_final_velocity[0][0]
                    # x[t, 3, 50] = player_final_velocity[1][0]

# fix the visualization problem for moving ego
            # test whether the transform of the deep learning model depends on the world frame or the ego's frame


            if n > 75:
                y = numpy.zeros((3, num))

                if n % 5 == 0:
                    for a in range(len(vehicles_list)):
                        actor = actor_list.find(vehicles_list[a])


                        actor_location = [[actor.get_location().x], [actor.get_location().y], [actor.get_location().z], [1]]
                        # actor_velocity = [[actor.get_velocity().x], [actor.get_velocity().y], [actor.get_velocity().z], [0]]

                        actor_final_location = numpy.dot(matrix, actor_location)
                        # actor_final_velocity = numpy.dot(matrix, actor_velocity)

                        y[0, a] = actor_final_location[0][0]
                        y[1, a] = actor_final_location[1][0]
                        y[2, a] = actor_final_location[2][0]
                        # y[2, a] = actor_final_velocity[0][0]
                        # y[3, a] = actor_final_velocity[1][0]

                    y[0, num-1] = player_final_location[0][0]
                    y[1, num-1] = player_final_location[1][0]
                    y[2, num-1] = player_final_location[2][0]
                    # y[2, 50] = player_final_velocity[0][0]
                    # y[3, 50] = player_final_velocity[1][0]
                    # update the (t, c, n) array where n is the vehicles around a distance 'd' from the ego
                    # get rid of velocity and add z to the position dimension of the (t, c, n) array
                    # test the inverse transform

                    x = numpy.insert(x, 15, y, 0)
                    x = numpy.delete(x, 0, 0)
                    matrix_all = numpy.insert(matrix_all, 15, inv_matrix, 0)
                    matrix_all = numpy.delete(matrix_all, 0, 0)


                    final_x = x
                    vehicle_num = num
                    i = vehicle_num-1
                    while i>=0:
                        if ((final_x[14, 0, i]-final_x[14, 0, vehicle_num-1])**2+(final_x[14, 1, i]-final_x[14, 1,
                                                                                                vehicle_num-1])**2)>=2500:
                            final_x = numpy.delete(final_x, i, 2)
                            vehicle_num = vehicle_num-1
                        i=i-1

                    #shape test
                    # print(x.shape)
                    # print(x.dtype)
                    numpy.set_printoptions(threshold=numpy.inf)
                    # print(final_x.shape)
                    # print(final_x)

                    # print(hist_vis.shape)



                    final_x = final_x.astype('float32')
                    # x = x.astype('float32')
                    # print(final_x.shape)
                    input = final_x.copy()

                    model = call_model()
                    preds = traj_pred(model, input)  # t c n





                    # print(inv_matrix.shape)
                    for i in range(final_x.shape[0]):

                        for j in range(final_x.shape[2]):
                            # if ((x[14, 0, j] - x[14, 0, num-1]) ** 2 + (
                            #         x[14, 1, j] - x[14, 1, num - 1]) ** 2) <= 2500:
                            pos_carla = numpy.array([float(final_x[i][0][j]), float(final_x[i][1][j]),
                                                 float(final_x[i][2][j]),
                                                 1.0])
                        # print(pos_carla.shape)

                            pos_carla = numpy.dot(matrix_all[i,:,:], pos_carla)
                            # print(pos_carla.shape)
                            #pos_carla = pos_carla[:3]
                            # print(pos_carla.shape)

                            pos_carla = carla.Location(float(pos_carla[0]), float(pos_carla[1]), float(pos_carla[2]))


                            world.debug.draw_point(pos_carla, size=0.05, color=carla.Color(r=0, g=255, b=0),
                                                   life_time=120)

                    for i in range(preds.shape[0]):

                        for j in range(preds.shape[2]):

                            pred_carla = numpy.array([float(preds[i][1][j]), float(preds[i][0][j]),
                                                     float(final_x[i][2][j]),
                                                     1.0])
                            # print(pos_carla.shape)

                            pred_carla = numpy.dot(inv_matrix, pred_carla)
                            # print(pos_carla.shape)
                            #pos_carla = pos_carla[:3]
                            # print(pos_carla.shape)

                            pred_carla = carla.Location(float(pred_carla[0]), float(pred_carla[1]), float(pred_carla[
                                                                                                              2]))


                            world.debug.draw_point(pred_carla, size=0.05, color=carla.Color(r=255, g=0, b=0),
                                                   life_time=120)
                    # # x: t c n
                    # preds = traj_pred(model, x) # t c n



                # update the (t, c, n) array where n is the vehicles around a distance 'd' from the ego
                # get rid of velocity and add z to the position dimension of the (t, c, n) array
                # test the inverse transform





                # if n % 75 == 0:
                #     n = 0


            #         u = actor_final_velocity[0]
            #         v = actor_final_velocity[1]
            #
            #         edgP1 = [ego_len / 2, ego_wid / 2]
            #         edgP2 = [-ego_len / 2, ego_wid / 2]
            #         edgP3 = [ego_len / 2, -ego_wid / 2]
            #         edgP4 = [-ego_len / 2, -ego_wid / 2]
            #         diagLen = 0.5 * (traffic_len ** 2 + traffic_wid ** 2) ** (1 / 2)
            #         angle = (actor.get_transform().rotation.yaw-player.get_transform().rotation.yaw)/180*(math.pi)
            #         angle2 = 0.5 * math.pi - angle
            #         angle3 = math.atan(traffic_wid / traffic_len)
            #         angle4 = math.pi - angle - angle3
            #         angle5 = 0.5 * math.pi
            #         octagon1 = [edgP1[0] + math.sin(angle4) * diagLen, edgP1[1] - math.cos(angle4) * diagLen]
            #         octagon2 = [octagon1[0] - math.sin(angle2) * traffic_wid, octagon1[1] + math.cos(angle2) * traffic_wid]
            #         octagon8 = [edgP1[0] + math.sin(angle4) * diagLen, edgP1[1] - math.cos(angle4) * diagLen - ego_wid]
            #         octagon3 = [octagon1[0] - math.sin(angle2) * traffic_wid - ego_len, octagon1[1] + math.cos(angle2) * traffic_wid]
            #         octagon4 = [-octagon8[0], -octagon8[1]]
            #         octagon5 = [octagon4[0], octagon4[1] - ego_wid]
            #         octagon6 = [-octagon2[0], -octagon2[1]]
            #         octagon7 = [-octagon2[0] + ego_len, -octagon2[1]]
            #         x = actor_final_location[0]
            #         y = actor_final_location[1]
            #         f = lambda y1, x1: 1 / (2 * math.pi * sigma ** 2) * math.exp(-0.5 * ((x1 - x) ** 2 / (sigma) ** 2 + (y1 - y) ** 2 / sigma ** 2))
            #         al1 = integrate.dblquad(f, octagon4[0], octagon6[0], getlinearline(octagon5, octagon6), getlinearline(octagon4, octagon3))
            #         al2 = integrate.dblquad(f, octagon6[0], octagon3[0], octagon6[1], getlinearline(octagon4, octagon3))
            #         al3 = integrate.dblquad(f, octagon3[0], octagon7[0], octagon6[1], octagon3[1])
            #         al4 = integrate.dblquad(f, octagon7[0], octagon2[0], getlinearline(octagon7, octagon8), octagon2[1])
            #         al5 = integrate.dblquad(f, octagon2[0], octagon1[0], getlinearline(octagon7, octagon8), getlinearline(octagon2, octagon1))
            #         all = [al1[0] + al2[0] + al3[0] + al4[0] + al5[0]]
            #         probability = round(all[0],4)
            #
            #
            #
            # print('id: ', vehicles_list[a], 'location:', actor_final_location[0][0], actor_final_location[1][0], actor_final_location[2][0], 'velocity:', actor_final_velocity[0][0], actor_final_velocity[1][0], actor_final_velocity[2][0])
            #         world.debug.draw_point(actor.get_location(), size=0.1, color=carla.Color(r=0, g=0, b=255), life_time=5)
            #         world.debug.draw_string(actor.get_location(), str(probability),  draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.5)

            #
            #
            # print(len(vehicles))










    finally:

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        print('\ndestroying %d vehicles' % len(vehicles_list))



        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])

        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        threads = []
        t1 = threading.Thread(target=main)
        threads.append(t1)
        t2 = threading.Thread(target=intention.my_intention)
        threads.append(t2)
        for t in threads:
            t.start()


    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')





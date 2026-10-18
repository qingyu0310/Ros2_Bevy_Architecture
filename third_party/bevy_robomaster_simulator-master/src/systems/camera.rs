use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use std::f32::consts::PI;

use crate::components::{
    CameraMode, Controlled, FollowingType, Infantry, InfantryGimbal, InfantryLaunchOffset,
    InfantryViewOffset, MainCamera,
};
use crate::config::SimulationConfig;
use crate::systems::ControllerState;

/// 鼠标捕获开关：默认捕获（转云台/转视角），Esc 释放给窗口系统
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
pub struct MouseCapture(pub bool);

/// 启动时捕获鼠标：锁定在窗口内且不可见，鼠标 delta 才能持续产生
pub fn init_mouse_capture(mut commands: Commands, mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>) {
    cursor.grab_mode = CursorGrabMode::Locked;
    cursor.visible = false;
    commands.insert_resource(MouseCapture(true));
}

/// Esc 在捕获与释放之间切换；释放后光标可见，可以操作其他窗口
pub fn toggle_mouse_capture(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut capture: ResMut<MouseCapture>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if !keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    capture.0 = !capture.0;
    if capture.0 {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    } else {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

pub fn following_controls(mut mode: ResMut<CameraMode>, controller: Res<ControllerState>) {
    if controller.controlled.switch_camera_just_pressed {
        mode.0 = match mode.0 {
            FollowingType::Free => FollowingType::Robot,
            FollowingType::Robot => FollowingType::ThirdPerson,
            FollowingType::ThirdPerson => FollowingType::Free,
        };
    }
}

pub fn update_camera_follow(
    camera_query: Single<(&mut Transform, &MainCamera), Without<Controlled>>,
    infantry: Single<&Transform, (With<Infantry>, With<Controlled>)>,
    gimbal: Single<&Transform, (With<Controlled>, With<InfantryGimbal>)>,
    view_offset: Single<&Transform, (With<Controlled>, With<InfantryViewOffset>)>,
    launch_offset: Single<&Transform, (With<Controlled>, With<InfantryLaunchOffset>)>,
    mode: Res<CameraMode>,
) {
    let gimbal_transform = gimbal.into_inner();
    let (mut camera_transform, camera_offset) = camera_query.into_inner();

    match mode.0 {
        FollowingType::Robot => {
            let view_offset_transform = view_offset.into_inner();
            let gimbal_world_rotation = infantry.rotation * gimbal_transform.rotation;
            let view_offset_world = gimbal_world_rotation * view_offset_transform.translation;

            camera_transform.translation = infantry.translation + view_offset_world;
            camera_transform.rotation = (gimbal_world_rotation
                * launch_offset.rotation
                * Quat::from_euler(EulerRot::ZYX, 0.0, 0.0, PI / 2.0))
            .normalize()
        }
        FollowingType::ThirdPerson => {
            // 相机高度与距离保持跟随偏移不变；环绕方向跟云台偏航，鼠标转云台时视角绕着车转。
            // 俯仰不参与，避免云台低头把相机带到地平面以下
            let base_transform = infantry.into_inner();
            let gimbal_world_rotation = base_transform.rotation * gimbal_transform.rotation;
            let (gimbal_yaw, _, _) = gimbal_world_rotation.to_euler(EulerRot::YXZ);
            let yaw_rotation = Quat::from_euler(EulerRot::YXZ, gimbal_yaw, 0.0, 0.0);

            camera_transform.translation =
                base_transform.translation + yaw_rotation * camera_offset.follow_offset;
            camera_transform.look_at(base_transform.translation, Vec3::Y);
        }
        FollowingType::Free => {}
    }
}

pub fn freecam_controls(
    time: Res<Time>,
    mode: Res<CameraMode>,
    config: Res<SimulationConfig>,
    mut mouse_motion_events: MessageReader<MouseMotion>,
    keyboard: Res<ButtonInput<KeyCode>>,
    camera_query: Single<&mut Transform, (With<MainCamera>, Without<Infantry>)>,
) {
    if mode.0 != FollowingType::Free {
        return;
    }

    let delta = time.delta_secs();
    let mut camera_transform = camera_query.into_inner();

    let mut mouse_delta = Vec2::ZERO;
    for event in mouse_motion_events.read() {
        mouse_delta += event.delta;
    }

    if mouse_delta != Vec2::ZERO {
        let (yaw, pitch, roll) = camera_transform.rotation.to_euler(EulerRot::YXZ);

        let new_yaw = yaw - mouse_delta.x * config.camera.mouse_sensitivity;
        let new_pitch = (pitch - mouse_delta.y * config.camera.mouse_sensitivity).clamp(-1.4, 1.4);

        camera_transform.rotation = Quat::from_euler(EulerRot::YXZ, new_yaw, new_pitch, roll);
    }

    let speed = config.camera.free_move_speed * delta;
    let forward = camera_transform.forward();
    let right = camera_transform.right();
    let up = camera_transform.up();

    if keyboard.pressed(KeyCode::KeyW) {
        camera_transform.translation += forward * speed;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        camera_transform.translation -= forward * speed;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        camera_transform.translation -= right * speed;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        camera_transform.translation += right * speed;
    }
    if keyboard.pressed(KeyCode::KeyN) {
        camera_transform.translation += up * speed;
    }
    if keyboard.pressed(KeyCode::KeyJ) {
        camera_transform.translation -= up * speed;
    }
}

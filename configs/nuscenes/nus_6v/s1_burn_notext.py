origin_h = 512
origin_w = 512
channel_num = 6 # means 6view here
nuscenes_image_path = '/home/nu_wq/'
layout_path_obs = "/root/paddlejob/workspace/layout/obs/"
layout_path_lane = "/root/paddlejob/workspace/layout/lane"

output_dir = "output_dd/nus_6v_nofz_3e-5_512/"

config = dict(
    # pretrained_model_path="./checkpoints/stable-diffusion-v1-4",
    pretrained_model_path="./checkpoints/stable-diffusion-v1-4_2D_pretrained",
    output_dir=output_dir,
    # resume_from_checkpoint="./outputs/bev_t_adapter/checkpoint-10000",
    arch_mode="{}view_img".format(channel_num), # custom img video finetune
    dataset="nuScenes",
    train_data_custom=dict(
        image_dir="/data/custom/data",
        label_dir="/data/custom/label",
        ori_h=origin_h,
        ori_w=origin_w,
        split='train',
        list_file=[],
        camera_names=["spherical_left_forward", "onsemi_obstacle", "spherical_right_forward", 
                      "spherical_right_backward", "spherical_backward", "spherical_left_backward"],
        transforms=None,
        with_params=True,
        undistorted=True,
        sample_rate=1,
        temporal=[12, 0],
        scene_label_path='',
        seq_len_1=1,
        seq_len_2=11
    ),
    train_data_nus=dict(
        dataroot=nuscenes_image_path,
        layout_path_obs=layout_path_obs,
        layout_path_lane=layout_path_lane,
        camera_names=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                      'CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT'],
        cond_mode='fuse',
        prompt="city road, car",
        dataset_set="train",
        use_caption=False,
        frame_num=channel_num,
        camid='CAM_FRONT_LEFT',
        image_height=origin_h,
        image_width=origin_w,
    ),
    validation_data=dict(
        prompts=["city road, car"],
        video_length=channel_num,
        height=origin_h,
        width=origin_w,
        num_inference_steps=20,
        guidance_scale=12.5,
        use_inv_latent=False,
        num_inv_steps=20,
        dataset_set="val"
    ),
    learning_rate=3e-5,
    train_batch_size=1,
    max_train_steps=500000000,
    checkpointing_steps=10000,
    validation_steps=200,
    trainable_modules=[
        "attn1.to_q", "attn2.to_q", "attn_temp", "conv_temporal", "skeleton",
    ],
    seed=33,
    mixed_precision='no',
    use_8bit_adam=False,
    gradient_checkpointing=False,
    enable_xformers_memory_efficient_attention=True,
    enable_text=False,
    prompt_mode="global", # global local both vqa
    burn_unet=True
)
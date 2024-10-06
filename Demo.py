import streamlit as st
import os
import cv2
from tqdm import tqdm
from glob import glob
import time
import tensorflow as tf
import numpy as np
from net import generator
from tools.GuidedFilter import guided_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_images(images, image_path, hw):
    images = (images.squeeze()+ 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, (hw[1], hw[0]))
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

def preprocessing(img, x8=True):
    h, w = img.shape[:2]
    if x8: # resize image to multiple of 8s
        def to_x8s(x):
            return 256 if x < 256 else x - x%8 # if using tiny model: x - x%16
        img = cv2.resize(img, (to_x8s(w), to_x8s(h)))
    return img/127.5 - 1.0

def load_test_data(image, x8=True):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = preprocessing(img, x8)
    img = np.expand_dims(img, axis=0)
    return img, image.shape[:2]

def sigm_out_scale(x):
    x = (x + 1.0) / 2.0
    return tf.clip_by_value(x, 0.0, 1.0)

def tanh_out_scale(x):
    x = (x - 0.5) * 2.0
    return tf.clip_by_value(x, -1.0, 1.0)

def test(checkpoint_dir, save_dir, image):
    result_dir = check_folder(save_dir)
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='AnimeGANv3_input')
    with tf.variable_scope("generator", reuse=False):
        _, _ = generator.G_net(test_real, True)
    with tf.variable_scope("generator", reuse=True):
        test_s0, test_m = generator.G_net(test_real, False)
        test_s1 = tanh_out_scale(guided_filter(sigm_out_scale(test_s0), sigm_out_scale(test_s0), 2, 0.01))

    variables = tf.contrib.framework.get_variables_to_restore()
    generator_var = [var for var in variables if var.name.startswith('generator') and ('main'  in var.name  or 'base'  in var.name) and 'Adam' not in var.name and 'support' not in var.name]
    saver = tf.train.Saver(generator_var)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        sess.run(tf.global_variables_initializer())
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else:
            print(" [*] Failed to find a checkpoint")
            return

        img, scale = load_test_data(image)
        real, s1, s0, m = sess.run([test_real, test_s1, test_s0, test_m], feed_dict={test_real: img})
        save_images(real, result_dir + '/a_result.png', scale)
        save_images(s1, result_dir + '/b_result.png', scale)
        save_images(s0, result_dir + '/c_result.png', scale)
        save_images(m, result_dir + '/d_result.png', scale)

        return result_dir + '/a_result.png', result_dir + '/b_result.png', result_dir + '/c_result.png', result_dir + '/d_result.png'

def main():
    st.title("AnimeGANv3 Demo")
    st.write("Upload an image to apply AnimeGANv3 style transfer.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    # Thêm menu để chọn style
    style_option = st.selectbox("Select a style:", ("Tabe", "Shinkai"))

    if uploaded_file is not None:
        image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Stylize"):
            with st.spinner('Processing...'):
                if style_option == "Tabe":
                    checkpoint_dir = 'checkpoint/generator_v3_Tabe_weight'
                else:
                    checkpoint_dir = 'checkpoint/generator_v3_Shinkai_weight'

                save_dir = 'style_results/'
                results = test(checkpoint_dir, save_dir, image)
                
                # Hiển thị kết quả song song
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(results[0], caption='Original Image', use_column_width=True)
                with col2:
                    st.image(results[3], caption='Stylized Image (m)', use_column_width=True)


if __name__ == '__main__':
    main()

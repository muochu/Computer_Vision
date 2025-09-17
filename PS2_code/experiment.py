import os
import cv2
import numpy as np
# from matplotlib import pyplot as plt

import ps2

INPUT_DIR = "input_images/"
OUTPUT_DIR = "output_images/"

# -- display utils
marker_color = (255, 0, 255)
text_color = (90, 90, 90)
text_thickness = 2
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5


def place_text(text, center, img, cache=None):
    if cache is None:
        cache = {}
    if "y_offset" in cache:
        cache["y_offset"] *= -1
    else:
        cache["y_offset"] = -30
    size = cv2.getTextSize(text, font_face, font_scale, text_thickness)
    y = center[1] + cache["y_offset"]
    if size[0][0] + center[0] > img.shape[1]:
        x = center[0] - size[0][0] - 5
    else:
        x = center[0] + 5
    cv2.rectangle(img, (x, y - size[0][1] - size[1]),
                  (x + size[0][0], y + size[0][1] - size[1]), (255, 255, 255),
                  cv2.FILLED)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color,
                text_thickness)


def draw_tl_center(image_in, center, state):
    # draw traffic light center and state
    center = (int(center[0]), int(center[1]))
    output = image_in
    cv2.drawMarker(output,
                   center,
                   marker_color,
                   markerType=cv2.MARKER_CROSS,
                   markerSize=11,
                   thickness=2)
    text = "(({}, {}), '{}')".format(center[0], center[1], state)
    place_text(text, center, output)
    return output


def mark_traffic_signs(image_in, signs_dict):
    # mark traffic sign centers
    output = image_in
    items = []
    for k, center in signs_dict.items():
        items.append((int(center[0]), k, center))
    items.sort()

    for _, k, center in items:
        center = (int(center[0]), int(center[1]))
        cv2.drawMarker(output,
                       center,
                       marker_color,
                       markerType=cv2.MARKER_CROSS,
                       markerSize=11,
                       thickness=2)
        text = "{}: ({}, {})".format(k, center[0], center[1])
        place_text(text, center, output)
    return output


def part_1a():

    input_images = ['scene_tl_1']
    output_labels = ['ps2-1-a-1']

    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):

        tl = cv2.imread("input_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite(os.path.join(OUTPUT_DIR,"{}.jpg".format(label)), img_out)


def part_1b():

    input_images = ['scene_constr_1']

    output_labels = ['ps2-1-b-1']

    sign_fns = [ps2.construction_sign_detection]

    sign_labels = ['construction']

    for img_in, label, fn, name in zip(input_images, output_labels, sign_fns,
                                       sign_labels):

        sign_img = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(sign_img)

        temp_dict = {name: coords}
        img_out = mark_traffic_signs(sign_img, temp_dict)
        cv2.imwrite(os.path.join(OUTPUT_DIR,"{}.jpg".format(label)), img_out)


def template_match_test():
    input_images = ['scene_tl_1', 'scene_constr_1', 'waldo_1']
    output_labels = ['ps2-2-a-1', 'ps2-2-b-1', 'ps2-2-c-1']
    templates = [
        'tl_template.png', 'construction_template.png', 'waldo1_template.png'
    ]
    for img_fl, img_template_fl, label in zip(
            input_images,
            templates,
            output_labels,
    ):
        img_in = cv2.imread("input_images/{}.png".format(img_fl))
        img_template = cv2.imread(
            "input_images/{}".format(img_template_fl))

        # restrict construction matching to orange regions
        if img_fl == 'scene_constr_1':
            hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
            orange_mask = cv2.inRange(hsv, (10,100,100), (25,255,255))
            img_for_matching = cv2.bitwise_and(img_in, img_in, mask=orange_mask)
        else:
            img_for_matching = img_in
        
        top_left = ps2.template_match(img_for_matching, img_template, "tm_nccor")
        th, tw = img_template.shape[:2]
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        im_out = img_in.copy()
        cv2.rectangle(im_out, top_left, bottom_right, (255, 255, 255), 2)
        text = "(({}, {}))".format(top_left[0], top_left[1])
        place_text(text, top_left, im_out)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "{}.jpg".format(label)), im_out)


def compression_runner():
    img_bgr = cv2.imread(INPUT_DIR + 'dog.jpg', cv2.IMREAD_COLOR)

    # compression thresholds
    thresholds = [0.1, 0.05, 0.001]
    
    for i, threshold in enumerate(thresholds, 1):
        img_compressed, compressed_frequency_img = ps2.compress_image_fft(
            img_bgr, threshold)
        
        # frequency visualization - shift and normalize properly
        eps = 1e-8
        freq_to_show = np.fft.fftshift(compressed_frequency_img, axes=(0, 1))
        mag_db = 20 * np.log10(np.maximum(eps, np.abs(freq_to_show)))
        # clip extreme values for better visualization
        mag_db_clipped = np.clip(mag_db, -100, 100)
        freq_vis = cv2.normalize(mag_db_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        img_compressed_uint8 = np.clip(img_compressed, 0, 255).astype(np.uint8)
        cv2.imwrite(OUTPUT_DIR + 'ps2-4-a-{}.jpg'.format(i), img_compressed_uint8)
        cv2.imwrite(OUTPUT_DIR + 'ps2-4-a-{}-freq.jpg'.format(i), freq_vis)


def low_pass_filter_runner():
    img_bgr = cv2.imread(INPUT_DIR + 'cat.jpg', cv2.IMREAD_COLOR)
    img_bgr = np.ndarray.astype(img_bgr, dtype=np.double)

    # filter radii - match spec requirements
    radii = [100, 50, 10]
    
    for i, radius in enumerate(radii, 1):
        img_low_pass, low_pass_frequency_img = ps2.low_pass_filter(
            img_bgr, radius)
        
        # frequency visualization - already shifted, just normalize
        eps = 1e-8
        freq_to_show = low_pass_frequency_img  # already centered
        mag_db = 20 * np.log10(np.maximum(eps, np.abs(freq_to_show)))
        # clip extreme values for better visualization
        mag_db_clipped = np.clip(mag_db, -100, 100)
        freq_vis = cv2.normalize(mag_db_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        img_low_pass_uint8 = np.clip(img_low_pass, 0, 255).astype(np.uint8)
        cv2.imwrite(OUTPUT_DIR + 'ps2-5-a-{}.jpg'.format(i), img_low_pass_uint8)
        cv2.imwrite(OUTPUT_DIR + 'ps2-5-a-{}-freq.jpg'.format(i), freq_vis)


if __name__ == "__main__":
    if not os.path.exists("output_images"):
        os.makedirs("output_images")
    part_1a()
    part_1b()
    template_match_test()
    compression_runner()
    low_pass_filter_runner()
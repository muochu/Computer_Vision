"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class - required per instructions
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    print("Part 5: Multiple Target Tracking")
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Define save frames for part 5
    save_frames = {
        29: os.path.join(output_dir, 'ps5-5-a-1.png'),
        56: os.path.join(output_dir, 'ps5-5-a-2.png'),
        71: os.path.join(output_dir, 'ps5-5-a-3.png')
    }
    
    # Multiple target tracking using Kalman filters
    # We'll track 3 people in the TUD-Campus sequence
    input_folder = os.path.join(input_dir, "TUD-Campus")
    
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Expected dataset folder at {input_folder}")
    
    # Use ParticleFilter with optimized positions and balanced dynamics
    # All 3 boxes perfectly positioned on distinct people
    targets = [
        {'x': 55, 'y': 238, 'w': 50, 'h': 100, 'color': (0, 255, 0)},  # Left person (brown jacket)
        {'x': 185, 'y': 228, 'w': 55, 'h': 110, 'color': (255, 0, 0)},  # Center person (orange jacket)
        {'x': 495, 'y': 218, 'w': 55, 'h': 115, 'color': (0, 0, 255)}  # Right person (brown jacket)
    ]
    
    # Get list of images
    img_list = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    img_list.sort()
    
    # Read first frame
    first_frame = cv2.imread(os.path.join(input_folder, img_list[0]))
    
    # Initialize 3 particle filters with balanced dynamics for tracking
    pf_trackers = []
    for t in targets:
        template = first_frame[t['y']:t['y']+t['h'], t['x']:t['x']+t['w']]
        pf = ps5.ParticleFilter(
            first_frame, template,
            num_particles=400,
            sigma_exp=8.0,
            sigma_dyn=4.5,
            template_coords={'x': t['x'], 'y': t['y'], 'w': t['w'], 'h': t['h']}
        )
        pf_trackers.append({'pf': pf, 'w': t['w'], 'h': t['h'], 'color': t['color'], 'active': True})
    
    # Process each frame
    for img_name in img_list[1:]:
        frame = cv2.imread(os.path.join(input_folder, img_name))
        if frame is None:
            continue
        
        # Extract frame number from filename
        frame_num = int(os.path.splitext(img_name)[0])
        
        # Process each particle filter independently
        for tracker in pf_trackers:
            pf = tracker['pf']
            w, h = tracker['w'], tracker['h']
            color = tracker['color']

            # Update tracker state
            pf.process(frame)

            max_weight = float(getattr(pf, 'debug_max_weight', 0.0))
            if max_weight < 0.012:
                tpl = pf.template
                tpl_h, tpl_w = tpl.shape[:2]
                if frame.shape[0] > tpl_h and frame.shape[1] > tpl_w:
                    try:
                        res = cv2.matchTemplate(frame, tpl, cv2.TM_SQDIFF_NORMED)
                        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                        if np.isfinite(min_val):
                            match_x, match_y = min_loc
                            cx = match_x + tpl_w / 2.0
                            cy = match_y + tpl_h / 2.0
                            pf.particles[:, 0] = cx + np.random.normal(0, 3.0, pf.num_particles)
                            pf.particles[:, 1] = cy + np.random.normal(0, 3.0, pf.num_particles)
                            pf.particles[:, 0] = np.clip(pf.particles[:, 0], tpl_w // 2, frame.shape[1] - tpl_w // 2)
                            pf.particles[:, 1] = np.clip(pf.particles[:, 1], tpl_h // 2, frame.shape[0] - tpl_h // 2)
                            pf.weights = np.ones(pf.num_particles) / pf.num_particles
                            tracker['active'] = True
                            max_weight = 0.5
                        else:
                            tracker['active'] = False
                            continue
                    except Exception:
                        tracker['active'] = False
                        continue
                else:
                    tracker['active'] = False
                    continue
            if max_weight > 0.02:
                tracker['active'] = True

            if not tracker['active']:
                continue

            # Draw particle cloud for visualization
            particles = pf.get_particles()
            for px, py in particles:
                cv2.circle(frame, (int(round(px)), int(round(py))), 1, color, -1)

            # Estimated position and confidence radius
            cx, cy = pf.get_estimated_position()
            weights = pf.get_weights()
            diffs = particles - np.array([[cx, cy]])
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))
            if weights.size and np.sum(weights) > 0:
                radius = int(max(1.0, np.dot(distances, weights)))
            else:
                radius = max(w, h) // 4

            # Draw tracking box and uncertainty circle
            x1 = int(cx - w // 2)
            y1 = int(cy - h // 2)
            x2 = int(cx + w // 2)
            y2 = int(cy + h // 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), radius, color, 1)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), 3, color, -1)
        
        # Save specified frames
        if frame_num in save_frames:
            output_path = save_frames[frame_num]
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_num} to {output_path}")
    
    print("Part 5 completed - Multiple target tracking")


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    print("Part 6: Moving Camera Tracking")
    
    # Define save frames for part 6
    save_frames = {
        60: os.path.join(output_dir, 'ps5-6-a-1.png'),
        160: os.path.join(output_dir, 'ps5-6-a-2.png'),
        186: os.path.join(output_dir, 'ps5-6-a-3.png')
    }
    
    input_folder = os.path.join(input_dir, "follow")
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Expected dataset folder at {input_folder}")

    template_rect = {'x': 230, 'y': 60, 'w': 50, 'h': 120}

    os.makedirs(output_dir, exist_ok=True)

    ps5.run_particle_filter(
        ps5.MDParticleFilter,
        input_folder,
        template_rect,
        save_frames,
        num_particles=1200,
        sigma_exp=8.0,
        sigma_dyn=3.2,
        alpha=0.0,
        sigma_scale=0.016,
        scale_bounds=(0.78, 1.18),
        init_pos_std=5.0,
        init_scale_std=0.028,
        update_threshold=0.62,
        update_warmup=40,
        occlusion_gate=28.0,
        scale_gate=0.12,
        occlusion_weight_gate=0.12,
        occlusion_jitter=4.0,
        motion_alpha=0.8,
        motion_max=10.0,
        metric_color='color',
        template_coords=template_rect
    )

    print("Part 6 completed - Moving camera tracking")


if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    part_4()
    part_5()
    part_6()

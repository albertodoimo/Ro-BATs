import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import scipy.signal as signal
import os
import imageio
try:
    from natsort import natsorted
    _natsort = True
except Exception:
    _natsort = False


def code(locations):
    case = 'audible'  # ultra or audible
    output_sig = 'sweep'  # noise or sweep

    order = 0 # Room reflection order
    distance = [1.8, 1.1]  # Source distance from array

    azimuth = np.atleast_1d(np.array(locations) / 180. * np.pi)  # Accepts scalar or list

    c = 343.
    fs = 48000
    nfft = 512
    if case == 'ultra':
        freq_range = [39000, 41000]
    elif case == 'audible':
        freq_range = [4000, 9000]
    else:
        raise ValueError('select a case')

    # snr_db = 1.
    # sigma2 = 10 ** (-snr_db / 10) / (4. * np.pi * distance) ** 2
    room_dim = np.r_[7., 4.5, 2.9]  # 3D room: x, y, z (z = 3 m)
    aroom = pra.ShoeBox(room_dim, fs=fs, max_order=order)

    center = room_dim / 2.0
    # Choose array type and build 3D microphone positions (shape (3, M))
    if case == 'ultra':
        M = 8
        d = 0.003
        # linear array along x centered at room center, at center z
        x = center[0] + (np.arange(M) - (M - 1) / 2.0) * d
        y = np.full(M, center[1])
        z = np.full(M, center[2])
        echo = np.vstack([x, y, z])
    elif case == 'audible':
        M = 6
        radius = 0.045
        phi0 = 0.0
        thetas = phi0 + 2.0 * np.pi * np.arange(M) / M
        x = center[0] + radius * np.cos(thetas)
        y = center[1] + radius * np.sin(thetas)
        z = np.full(M, room_dim[2]*0.08)
        echo = np.vstack([x, y, z])
    else:
        raise ValueError('select a case')

    aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))

    # Signal
    if output_sig == 'sweep':
        tone_durn = 10e-3
        t_tone = np.linspace(0, tone_durn, int(fs * tone_durn))
        if case == 'ultra':
            chirp = signal.chirp(t_tone, 40e3, t_tone[-1], 40e3)
        elif case == 'audible':
            chirp = signal.chirp(t_tone, 20e3, t_tone[-1], 4e3)
        chirp *= signal.windows.hann(chirp.size)
        source_signal = chirp
    elif output_sig == 'noise':
        rng = np.random.RandomState(23)
        duration_samples = int(fs)
        source_signal = rng.randn(duration_samples)
    else:
        raise ValueError('select an output_sig')

    # # Plot the chirp signal
    # plt.figure(figsize=(10, 4))
    # plt.plot(t_tone * 1000, chirp)  # Convert time to milliseconds
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Amplitude')
    # plt.title('Chirp Signal')
    # plt.grid(True)
    # plt.show()
    # # plt.savefig('chirp_signal.png')


    # Add multiple sources together (place sources in the horizontal plane at center z)
    sources = []
    for i, ang in enumerate(azimuth):
        # place source at given azimuth and distance, but match microphone array z
        xy = center[:2] + distance[i] * np.r_[np.cos(ang), np.sin(ang)]
        # assume all microphones share the same z; fall back to room center if not available
        try:
            z_src = float(echo[2, 0]*1.5)
        except Exception:
            z_src = float(center[2])
        source_location = np.r_[xy, z_src]
        aroom.add_source(source_location, source_signal)
        sources.append(source_location)
        print(f"Source {i + 1} location: {source_location}")

    aroom.simulate()

    # # Plot 3D room layout
    # fig = plt.figure(figsize=(12, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # aroom.plot(fig=fig, ax=ax)
    # ax.set_xlim([-1, room_dim[0] + 1])
    # ax.set_ylim([-1, room_dim[1] + 1])
    # ax.set_zlim([-1, room_dim[2] + 1])
    # plt.title('Room layout (3D)', fontdict={'fontsize': 15})

    # # Save only initial room configuration (avoid saving on every frame)
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # room_shape_path = os.path.join(script_dir, f'room_shape.jpg')
    # plt.savefig(room_shape_path, dpi=200, bbox_inches='tight')
    # plt.close(fig)

    # DOA estimation
    X = pra.transform.stft.analysis(aroom.mic_array.signals.T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    algo_names = ['SRP', 'MUSIC', 'TOPS']
    spatial_resp = dict()
    doa_results = dict()
    for algo_name in algo_names:
        doa = pra.doa.algorithms[algo_name](echo, fs, nfft, c=c, num_src=len(azimuth), max_four=4)
        doa.locate_sources(X, freq_range=freq_range)
        doa_results[algo_name] = doa.azimuth_recon  # list of estimated azimuths

        if hasattr(doa, 'grid') and hasattr(doa.grid, 'values'):
            resp = np.asarray(doa.grid.values)
        elif hasattr(doa, 'pseudo_spectrum'):
            resp = np.asarray(doa.pseudo_spectrum)
        elif hasattr(doa, 'spectrum'):
            resp = np.asarray(doa.spectrum)
        elif hasattr(doa, 'spatial_spectrum'):
            resp = np.asarray(doa.spatial_spectrum)
        else:
            try:
                n_angles = doa.grid.azimuth.size
            except Exception:
                n_angles = 360
            resp = np.zeros(n_angles)
        # normalize
        mn = resp.min()
        mx = resp.max()
        spatial_resp[algo_name] = (resp - mn) / (mx - mn) if mx > mn else np.zeros_like(resp)

    # Polar plot
    base = 1.0
    height = 10.0
    az = np.atleast_1d(azimuth)
    try:
        phi_plt = doa.grid.azimuth
    except Exception:
        phi_plt = np.linspace(0, 2 * np.pi, num=spatial_resp[algo_names[0]].size, endpoint=False)
    phi_plt = np.atleast_1d(phi_plt)
    fig = plt.figure(figsize=(14, 7))
    for i, algo_name in enumerate(algo_names, 1):
        ax = fig.add_subplot(2, 3, i, projection='polar')
        c_phi_plt = np.r_[phi_plt, phi_plt[0]]
        c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
        ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=2, alpha=0.55,
                linestyle='-', label="spatial\nspectrum")
        plt.title(algo_name, fontdict={'fontsize': 15}, loc='left')

        # plot true source locations
        ax.scatter(az, base + height * np.ones(len(az)), c='k', s=200, marker='*', label='true\nlocations')

        # plot estimated source locations
        est_azimuth = doa_results[algo_name]
        ax.scatter(est_azimuth, base + height * np.ones(len(est_azimuth)), c='r', s=80, marker='o', label='estimated\nlocations')

        ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
        ax.xaxis.set_label_coords(0.5, -0.11)
        ax.set_yticks(np.linspace(0, 1, 2))
        ax.set_ylim([0, 1.05 * (base + height)])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8, bbox_to_anchor=(1.5, 0.6))

    # Save frame
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frames_dir = os.path.join(script_dir, f'frames_{case}')
    os.makedirs(frames_dir, exist_ok=True)
    frame_idx = int(round(locations[1])) if np.ndim(locations) else int(round(locations))
    frame_path = os.path.join(frames_dir, f'frame_{frame_idx:03d}.png')
    plt.savefig(frame_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # GIF
    filenames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    if _natsort:
        filenames = natsorted(filenames)
    else:
        filenames = sorted(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.v2.imread(os.path.join(frames_dir, filename)))
    gif_path = os.path.join(script_dir, f'animation_{case}_3d_{freq_range[0]}_{freq_range[1]}_order_{order}_output_{output_sig}.gif')
    if images:
        imageio.mimsave(gif_path, images, duration=0.25)

    return frame_path, gif_path


if __name__ == "__main__":

    n_frames = 36

    # Static source angle in degrees (fixed)
    static_angle = 35.0

    print(f"Animating 1 static source and 1 moving source for {n_frames} frames...")
    for f in range(n_frames):
        # Moving source angle rotates from 0 to 360°
        moving_angle = (f * (360.0 / n_frames)) % 360.0
        angles = [static_angle, moving_angle]

        print(f"Frame {f + 1}/{n_frames}: static = {static_angle:.1f}°, moving = {moving_angle:.1f}°")
        code(angles)

import numpy as np


def randomize2d(data):
    K, T = data.shape
    A = np.fft.fft(data)
    phases = np.angle(A)
    phases_lh = np.random.uniform(0, 2 * np.pi,
                                  T // 2
                                  if T % 2 == 1
                                  else T // 2 - 1)
    phases_lh = np.array(K * [phases_lh])
    phases_rh = -phases_lh[:, ::-1]
    if T % 2 == 0:
        phases = np.hstack((phases[:, 0, None],
                            phases_lh,
                            phases[:, phases.shape[1] // 2,
                                   None],
                            phases_rh))
    else:
        phases = np.hstack((phases[:, 0, None],
                            phases_lh,
                            phases_rh))
    A_new = A * np.exp(1j * phases)
    return np.fft.ifft(A_new).real

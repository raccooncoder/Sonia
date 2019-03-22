import numpy as np

def transpose_range(samples):
	merged_sample = np.zeros_like(samples[0])

	for sample in samples:
		merged_sample = np.maximum(merged_sample, sample)

	merged_sample = np.amax(merged_sample, axis=0)
	min_note = np.argmax(merged_sample)
	max_note = merged_sample.shape[0] - np.argmax(merged_sample[::-1])

	return min_note, max_note

def generate_add_centered_transpose(samples):
	num_notes = samples[0].shape[1]
	min_note, max_note = transpose_range(samples)
	s = num_notes // 2 - (max_note + min_note) // 2
	out_samples = samples
	out_lens = [len(samples), len(samples)]

	for i in range(len(samples)):
		out_sample = np.zeros_like(samples[i])
		out_sample[:, min_note + s : max_note + s] = samples[i][:, min_note : max_note]
		out_samples.append(out_sample)

	return out_samples, out_lens

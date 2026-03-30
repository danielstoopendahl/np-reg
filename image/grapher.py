from pathlib import Path
import csv
from statistics import median

import matplotlib.pyplot as plt


def read_losses(csv_path: Path):
	epochs = []
	train_losses = []
	test_losses = []

	with csv_path.open("r", newline="") as file:
		reader = csv.DictReader(file)
		for row in reader:
			epochs.append(int(row["epoch"]))
			train_losses.append(float(row["train_loss"]))
			test_losses.append(float(row["test_loss"]))

	return epochs, train_losses, test_losses


def remove_upward_spikes(values, window_radius=3, spike_ratio=2.0, min_increase=1.5):
	"""Replace extreme local upward spikes using a neighborhood median."""
	if len(values) < 3:
		return values[:]

	cleaned = values[:]
	for index, current in enumerate(values):
		left = max(0, index - window_radius)
		right = min(len(values), index + window_radius + 1)
		neighborhood = values[left:index] + values[index + 1:right]

		if not neighborhood:
			continue

		local_median = median(neighborhood)
		if current > local_median * spike_ratio and (current - local_median) > min_increase:
			cleaned[index] = float(local_median)

	return cleaned


def main():
	root = Path(__file__).resolve().parent
	data_dir = root / "rankgraphs"

	model_files = {
		"Vanilla": data_dir / "vanilla_losses.csv",
		"Dropout": data_dir / "dropout_losses.csv",
		"Norm-preserving": data_dir / "normperserving_losses.csv",
		"GSVD norm-preserving": data_dir / "gsvd_normperserving_losses.csv",
	}

	fig, ax = plt.subplots(figsize=(12, 7))
	color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

	for idx, (model_name, file_path) in enumerate(model_files.items()):
		epochs, train_loss, test_loss = read_losses(file_path)
		train_loss = remove_upward_spikes(train_loss)
		test_loss = remove_upward_spikes(test_loss)
		color = color_cycle[idx % len(color_cycle)]

		ax.plot(
			epochs,
			train_loss,
			linestyle="-",
			color=color,
			linewidth=2,
			label=f"{model_name} train",
		)
		ax.plot(
			epochs,
			test_loss,
			linestyle="--",
			color=color,
			linewidth=2,
			label=f"{model_name} test",
		)

	ax.set_title("Training and Test Loss Comparison")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.grid(True, alpha=0.3)
	ax.legend(ncol=2)
	fig.tight_layout()

	output_path = data_dir / "combined_losses.png"
	fig.savefig(output_path, dpi=200)
	plt.show()


if __name__ == "__main__":
	main()

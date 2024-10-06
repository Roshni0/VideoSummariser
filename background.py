# This code identifies 3-5 key frames
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function to extract frames from a video
def extract_frames(video_path, frame_rate):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames = []
    count = 0
    while success:
        if count % frame_rate == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames

# Function to calculate the HSV histogram of a frame
def calculate_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to select keyframes based on histogram differences
def select_keyframes(frames, num_keyframes=5):
    histograms = [calculate_histogram(frame) for frame in frames]
    differences = []
    for i in range(1, len(histograms)):
        diff = np.linalg.norm(histograms[i] - histograms[i - 1])
        differences.append((diff, i))
    differences.sort(reverse=True)
    keyframe_indices = [i for _, i in differences[:num_keyframes]]
    keyframes = [frames[i] for i in keyframe_indices]
    return keyframes

# Function to select keyframes using KMeans clustering
def kmeans_keyframes(frames, num_keyframes=5):
    histograms = [calculate_histogram(frame) for frame in frames]
    kmeans = KMeans(n_clusters=num_keyframes, random_state=0).fit(histograms)
    keyframes = []
    for cluster_idx in range(num_keyframes):
        cluster_frames = np.where(kmeans.labels_ == cluster_idx)[0]
        keyframes.append(frames[cluster_frames[0]])
    return keyframes

# Function to display keyframes using matplotlib
def display_frames(frames, num_frames=5):
    plt.figure(figsize=(10, 5))
    for i in range(min(num_frames, len(frames))):  # Show up to num_frames frames
        plt.subplot(1, num_frames, i + 1)
        plt.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.axis('off')
    plt.show()

# Path to video and frame rate
video_path = 'goals.mp4'
frame_rate = 30  

# Extract frames from the video
frames = extract_frames(video_path, frame_rate)

# Select keyframes based on histogram differences (choose your method)
keyframes_histogram = select_keyframes(frames, num_keyframes=5)

# Select keyframes based on KMeans clustering
keyframes_kmeans = kmeans_keyframes(frames, num_keyframes=5)

# Visually
print("Keyframes selected using Histogram Diff:")
display_frames(keyframes_histogram, num_frames=5)

print("Keyframes selected using KMeans Clusteringg:")
display_frames(keyframes_kmeans, num_frames=5)

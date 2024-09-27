from flask import Flask, jsonify, request, render_template, session
import random
import math

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# KMeans state storage
kmeans_state = {
    'data': [],
    'centroids': [],
    'clusters': [],
    'k': 0,
    'iteration': 0
}

# Function to generate random 2D data points
def generate_random_data(num_points=50, x_range=(0, 100), y_range=(0, 100)):
    data = [[random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])] for _ in range(num_points)]
    return data

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def kmeans_single_step(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    
    # Assign points to the closest centroid
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_index = distances.index(min(distances))
        clusters[closest_index].append(point)
    
    # Recompute the centroids as the mean of each cluster
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Avoid division by zero
            new_centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(random.choice(data))  # In case of empty cluster
    
    return new_centroids, clusters

def kmeans_final(data, k, initial_centroids, max_iterations=100):
    centroids = initial_centroids[:]
    for _ in range(max_iterations):
        new_centroids, clusters = kmeans_single_step(data, centroids)
        if new_centroids == centroids:
            break  # Convergence
        centroids = new_centroids
    return centroids, clusters

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    global kmeans_state
    data = generate_random_data()
    kmeans_state['data'] = data
    return jsonify(data)

@app.route('/initialize_kmeans', methods=['POST'])
def initialize_kmeans():
    global kmeans_state
    data = request.json['data']
    k = request.json['k']
    initialization_method = request.json['initialization']
    
    if initialization_method == 'random':
        initial_centroids = random.sample(data, k)
    elif initialization_method == 'farthest':
        initial_centroids = [random.choice(data)]
        for _ in range(1, k):
            farthest_point = max(data, key=lambda point: min(euclidean_distance(point, c) for c in initial_centroids))
            initial_centroids.append(farthest_point)
    elif initialization_method == 'kmeans++':
        initial_centroids = [random.choice(data)]
        for _ in range(1, k):
            distances = [min(euclidean_distance(point, c) for c in initial_centroids) for point in data]
            weighted_probabilities = [d / sum(distances) for d in distances]
            next_centroid = random.choices(data, weights=weighted_probabilities)[0]
            initial_centroids.append(next_centroid)
    elif initialization_method == 'manual':
        initial_centroids = request.json['initial_centroids']  # Expecting a list of coordinates

    # Initialize state for step-by-step
    kmeans_state['centroids'] = initial_centroids
    kmeans_state['clusters'] = [[] for _ in range(k)]
    kmeans_state['data'] = data
    kmeans_state['k'] = k
    kmeans_state['iteration'] = 0
    
    return jsonify({
        'centroids': kmeans_state['centroids'],
        'clusters': kmeans_state['clusters']
    })

@app.route('/step_kmeans', methods=['POST'])
def step_kmeans():
    global kmeans_state
    
    centroids = kmeans_state['centroids']
    data = kmeans_state['data']
    
    new_centroids, clusters = kmeans_single_step(data, centroids)
    
    # Update state
    kmeans_state['centroids'] = new_centroids
    kmeans_state['clusters'] = clusters
    kmeans_state['iteration'] += 1
    
    return jsonify({
        'centroids': new_centroids,
        'clusters': clusters
    })

@app.route('/run_kmeans_final', methods=['POST'])
def run_kmeans_final():
    data = request.json['data']
    k = request.json['k']
    initial_centroids = request.json['initial_centroids']
    
    centroids, clusters = kmeans_final(data, k, initial_centroids)
    return jsonify({
        'centroids': centroids,
        'clusters': clusters
    })

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)


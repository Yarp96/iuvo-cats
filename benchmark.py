import requests
import time
import statistics
import concurrent.futures
import argparse
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def send_request(api_url: str, image_url: str) -> Dict[str, Any]:
    """Send a single request to the API and return response with timing info."""
    payload = {"image_url": image_url}
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    response = requests.post(api_url, json=payload, headers=headers)
    end_time = time.time()
    
    response_time = end_time - start_time
    status_code = response.status_code
    
    result = {
        "response_time": response_time,
        "status_code": status_code,
        "success": status_code == 200
    }
    
    if status_code == 200:
        result["response_data"] = response.json()
    else:
        result["error"] = response.text
        
    return result

def run_benchmark(api_url: str, image_url: str, num_requests: int, concurrency: int) -> List[Dict[str, Any]]:
    """Run benchmark with specified number of requests and concurrency level."""
    results = []
    
    with tqdm(total=num_requests, desc=f"Sending {num_requests} requests (concurrency={concurrency})") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request, api_url, image_url) for _ in range(num_requests)]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    
    return results

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze benchmark results and return statistics."""
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    if not results:
        return {"error": "No results to analyze"}
    
    response_times = [r["response_time"] for r in successful_requests]
    
    if not response_times:
        return {
            "total_requests": len(results),
            "successful_requests": 0,
            "failed_requests": len(failed_requests),
            "error": "No successful requests to analyze"
        }
    
    total_time = max(sum(response_times), 0.001)  # Avoid division by zero
    
    analysis = {
        "total_requests": len(results),
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "total_time": total_time,
        "throughput": len(successful_requests) / total_time,
        "min_response_time": min(response_times) if response_times else None,
        "max_response_time": max(response_times) if response_times else None,
        "avg_response_time": statistics.mean(response_times) if response_times else None,
        "median_response_time": statistics.median(response_times) if response_times else None,
        "p95_response_time": np.percentile(response_times, 95) if response_times else None,
        "p99_response_time": np.percentile(response_times, 99) if response_times else None
    }
    
    if failed_requests:
        error_codes = {}
        for req in failed_requests:
            status = req["status_code"]
            error_codes[status] = error_codes.get(status, 0) + 1
        analysis["error_codes"] = error_codes
    
    return analysis

def plot_results(results: List[Dict[str, Any]], output_file: str = None):
    """Generate plots for benchmark results."""
    successful_requests = [r for r in results if r["success"]]
    response_times = [r["response_time"] for r in successful_requests]
    
    if not response_times:
        print("No successful requests to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Response time histogram
    ax1.hist(response_times, bins=30, alpha=0.7, color='blue')
    ax1.set_title('Response Time Distribution')
    ax1.set_xlabel('Response Time (seconds)')
    ax1.set_ylabel('Frequency')
    
    # Response time over requests
    ax2.plot(range(len(response_times)), sorted(response_times), 'r-')
    ax2.set_title('Response Time (sorted)')
    ax2.set_xlabel('Request #')
    ax2.set_ylabel('Response Time (seconds)')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Benchmark Cat Facial Landmark Detection API')
    parser.add_argument('--url', type=str, required=True, help='API endpoint URL')
    parser.add_argument('--image', type=str, default="https://media.istockphoto.com/id/157671964/photo/portrait-of-a-tabby-cat-looking-at-the-camera.jpg?s=612x612&w=0&k=20&c=iTsJO6vuQ5w3hL5pWn42C91ziMRUsYd725oUGRRewjM=", 
                        help='Image URL to test with')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests to send')
    parser.add_argument('--concurrency', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--plot', type=str, help='Output file for plot (PNG)')
    
    args = parser.parse_args()
    
    print(f"Starting benchmark against {args.url}")
    print(f"Image URL: {args.image}")
    print(f"Sending {args.requests} requests with concurrency level {args.concurrency}")
    
    results = run_benchmark(args.url, args.image, args.requests, args.concurrency)
    analysis = analyze_results(results)
    
    print("\nBenchmark Results:")
    print(f"Total Requests: {analysis['total_requests']}")
    print(f"Successful Requests: {analysis['successful_requests']}")
    print(f"Failed Requests: {analysis['failed_requests']}")
    
    if analysis['successful_requests'] > 0:
        print(f"\nResponse Time Statistics:")
        print(f"Min: {analysis['min_response_time']:.4f} seconds")
        print(f"Max: {analysis['max_response_time']:.4f} seconds")
        print(f"Average: {analysis['avg_response_time']:.4f} seconds")
        print(f"Median: {analysis['median_response_time']:.4f} seconds")
        print(f"95th Percentile: {analysis['p95_response_time']:.4f} seconds")
        print(f"99th Percentile: {analysis['p99_response_time']:.4f} seconds")
        
        print(f"\nThroughput: {analysis['throughput']:.2f} requests/second")
    
    if analysis.get('error_codes'):
        print("\nError Codes:")
        for code, count in analysis['error_codes'].items():
            print(f"  {code}: {count} requests")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    if args.plot:
        plot_results(results, args.plot)
    elif analysis['successful_requests'] > 0:
        plot_results(results)

if __name__ == "__main__":
    main()

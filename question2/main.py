from perplexity_analysis import run_perplexity_analysis
from sampling_comparison import run_sampling_comparison

def main():
    print("Running perplexity analysis...")
    run_perplexity_analysis()
    print("Perplexity analysis complete. Results saved to results/perplexity_results.txt")
    
    print("\nRunning sampling comparison...")
    run_sampling_comparison()
    print("Sampling comparison complete. Results saved to results/sampling_results.txt")
    
    print("\nAll analyses complete!")

if __name__ == "__main__":
    main()
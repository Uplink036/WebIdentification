cpus=$(top -b -n 1 | grep cpu | wc -l)
cpu_limit=0.8
limited_cpus=$(echo "$cpus * $cpu_limit" | bc)
echo "$limited_cpus"

docker run --rm \
    --net=host \
    --cpus=$limited_cpus \
	-v $(pwd)/fetched_data:/app/CV_WebIdentification \
	--env URI=bolt://localhost:7687 \
	--env USERNAME=neo4j \
	--env PASSWORD=password \
	webidentification_data_fetcher:latest

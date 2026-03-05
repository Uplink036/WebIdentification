cpus=$(top -b -n 1 | grep cpu | wc -l)
echo "$cpus"
cpu_limit=0.8
echo "$cpu_limit"
limited_cpus=$(echo "$cpus * $cpu_limit" | bc)
echo "$limited_cpus"

docker run --rm \
    --net=host \
    --cpus=$limited_cpus \
	--env URI=bolt://localhost:7687 \
	--env USERNAME=neo4j \
	--env PASSWORD=password \
	webidentification_data_loader:latest
    
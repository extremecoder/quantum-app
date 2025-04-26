# Use the existing image as the base
FROM abhishekt/quantum-microservice:latest

# Define the path to the default circuit file on the host
ARG DEFAULT_CIRCUIT_FILE=my_circuit.qasm

# Define the target directory inside the container
ARG TARGET_DIR=/microservice/circuits/default

# Create the target directory if it doesn't exist
# The WORKDIR in the base image is /microservice, so paths can be relative
RUN mkdir -p circuits/default

# Copy the specified default circuit file from the build context
# into the target directory inside the container
COPY ${DEFAULT_CIRCUIT_FILE} ${TARGET_DIR}/
# Export the port as an environment variable (don't override CMD)
ENV PORT=8889
EXPOSE 8889

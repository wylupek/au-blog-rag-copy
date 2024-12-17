# Use a base image
FROM python:3.11-slim

# Install necessary packages
RUN apk update && apk add --no-cache curl jq bash coreutils util-linux

# Set the working directory
WORKDIR /app

# Copy scripts
COPY infakt/data-scrap.sh /app/data-scrap.sh
COPY infakt/infakt-clients.sh /app/infakt-clients.sh
COPY infakt/infakt-invoices.sh /app/infakt-invoices.sh

# Set the script as executable
RUN chmod +x /app/data-scrap.sh
RUN chmod +x /app/infakt-clients.sh 
RUN chmod +x /app/infakt-invoices.sh

# Run the script
CMD ["./data-scrap.sh"]
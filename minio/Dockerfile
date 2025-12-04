FROM minio/minio:latest

# Create data directory
RUN mkdir -p /data

# Copy the startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose MinIO API port (Render will map this)
EXPOSE 9000

# Run the startup script
CMD ["/start.sh"]
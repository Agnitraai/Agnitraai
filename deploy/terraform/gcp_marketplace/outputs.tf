output "service_url" {
  description = "Public HTTPS endpoint for the marketplace runtime."
  value       = google_cloud_run_service.runtime.status[0].url
}

output "usage_endpoint" {
  description = "Convenience URL for the /usage metering endpoint."
  value       = "${google_cloud_run_service.runtime.status[0].url}/usage"
}

output "service_account_email" {
  description = "Service account executing the runtime container."
  value       = google_service_account.runtime.email
}

output "service_account_secret" {
  description = "Secret Manager resource holding the service account key."
  value       = google_secret_manager_secret.sa.id
}

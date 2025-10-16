output "ingress_fqdn" {
  description = "Public FQDN for the container app."
  value       = azurerm_container_app.runtime.ingress[0].fqdn
}

output "usage_endpoint" {
  description = "HTTPS endpoint for the marketplace usage API."
  value       = "https://${azurerm_container_app.runtime.ingress[0].fqdn}/usage"
}

output "identity_principal_id" {
  description = "Managed identity principal ID assigned to the container app."
  value       = azurerm_container_app.runtime.identity[0].principal_id
}

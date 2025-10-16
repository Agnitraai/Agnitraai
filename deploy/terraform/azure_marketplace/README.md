# Azure Marketplace Deployment Module

This Terraform module provisions the infrastructure required to run the Agnitra
Marketplace runtime on Azure Container Apps. It configures the container with
the Azure Marketplace metering environment variables so that the `/usage`
endpoint can submit events through the Azure SaaS metering API.

Resources created:

- Resource group, Log Analytics workspace, and Container Apps environment.
- Container App with system-assigned managed identity, autoscaling rules, and
  public ingress.
- Secrets injected into the container that carry the Azure AD client secret
  used for marketplace metering authentication.

## Usage

```hcl
module "agnitra_marketplace" {
  source = "./deploy/terraform/azure_marketplace"

  resource_group_name      = "rg-agnitra-marketplace"
  location                 = "eastus"
  container_image          = "ghcr.io/agnitra/marketplace-runtime:latest"
  marketplace_resource_id  = "/subscriptions/.../resourceGroups/.../providers/Microsoft.SaaS/saasresources/agnitra"
  marketplace_plan_id      = "agnitra-production"
  tenant_id                = "00000000-0000-0000-0000-000000000000"
  client_id                = "11111111-1111-1111-1111-111111111111"
  client_secret            = var.azure_marketplace_client_secret
}
```

The module outputs the public FQDN that exposes the `/usage` endpoint; register
this endpoint with the Azure Marketplace SaaS metering integration.

output "load_balancer_dns" {
  description = "Public DNS name for the Application Load Balancer."
  value       = aws_lb.this.dns_name
}

output "service_name" {
  description = "ECS service name handling the Agnitra runtime."
  value       = aws_ecs_service.this.name
}

output "usage_endpoint" {
  description = "Marketplace usage endpoint exposed by the runtime."
  value       = "http://${aws_lb.this.dns_name}/usage"
}

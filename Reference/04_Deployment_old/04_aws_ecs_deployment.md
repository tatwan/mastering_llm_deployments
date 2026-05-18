# ☁️ AWS ECS Deployment with Terraform

**Module 04 | Guide 4 of 4**

Deploy your containerized ML applications to AWS ECS (Elastic Container Service) using Infrastructure as Code.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AWS Cloud                                  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                        VPC                                     │  │
│  │                                                                │  │
│  │  ┌─────────────────┐         ┌─────────────────┐               │  │
│  │  │  Public Subnet  │         │ Private Subnet  │               │  │
│  │  │                 │         │                 │               │  │
│  │  │  ┌───────────┐  │         │  ┌───────────┐  │               │  │
│  │  │  │    ALB    │──┼─────────┼─→│    ECS    │  │               │  │
│  │  │  │  (Load    │  │         │  │  Cluster  │  │               │  │
│  │  │  │ Balancer) │  │         │  │           │  │               │  │
│  │  │  └───────────┘  │         │  │ ┌───────┐ │  │               │  │
│  │  │                 │         │  │ │Task 1 │ │  │               │  │
│  │  └─────────────────┘         │  │ │Task 2 │ │  │               │  │
│  │                              │  │ │Task N │ │  │               │  │
│  │                              │  │ └───────┘ │  │               │  │
│  │                              │  └───────────┘  │               │  │
│  │                              └─────────────────┘               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────┐                                                  │
│  │      ECR       │  (Container Registry)                            │
│  │   my-ml-app    │                                                  │
│  └────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Terraform** installed
4. **Docker** installed

### Installing Terraform

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Verify
terraform --version
```

---

## Step 1: Push Image to ECR

### Create ECR Repository

```bash
# Create repository
aws ecr create-repository \
    --repository-name my-ml-app \
    --region us-east-1

# Get repository URI
aws ecr describe-repositories --repository-names my-ml-app
```

### Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag my-ml-app:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:latest

# Push
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:latest
```

---

## Step 2: Terraform Configuration

### Project Structure

```
terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── ecs.tf
├── alb.tf
└── terraform.tfvars
```

### variables.tf

```hcl
variable "aws_region" {
  default = "us-east-1"
}

variable "app_name" {
  default = "my-ml-app"
}

variable "container_port" {
  default = 8000
}

variable "container_image" {
  description = "ECR image URI"
}

variable "cpu" {
  default = 256
}

variable "memory" {
  default = 512
}

variable "desired_count" {
  default = 2
}

variable "vpc_id" {
  description = "VPC ID for deployment"
}

variable "subnet_ids" {
  description = "Subnet IDs for ECS tasks"
  type        = list(string)
}
```

### main.tf

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.app_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Task Execution Role
resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.app_name}-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
```

### ecs.tf

```hcl
# Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = var.app_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([
    {
      name  = var.app_name
      image = var.container_image
      
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 120
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/${var.app_name}"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      environment = [
        {
          name  = "MODEL_NAME"
          value = "distilbert-base-uncased-finetuned-sst-2-english"
        }
      ]
    }
  ])
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 7
}

# Security Group for ECS Tasks
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.app_name}-ecs-tasks"
  description = "Allow traffic to ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ECS Service
resource "aws_ecs_service" "app" {
  name            = var.app_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.app_name
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.app]
}
```

### alb.tf

```hcl
# Security Group for ALB
resource "aws_security_group" "alb" {
  name        = "${var.app_name}-alb"
  description = "Allow HTTP traffic"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Application Load Balancer
resource "aws_lb" "app" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.subnet_ids
}

# Target Group
resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
    timeout             = 30
    interval            = 60
    matcher             = "200"
  }
}

# Listener
resource "aws_lb_listener" "app" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
```

### outputs.tf

```hcl
output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.app.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = var.container_image
}

output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}
```

### terraform.tfvars (Example)

```hcl
aws_region      = "us-east-1"
app_name        = "my-ml-app"
container_image = "123456789.dkr.ecr.us-east-1.amazonaws.com/my-ml-app:latest"
vpc_id          = "vpc-xxxxxxxxx"
subnet_ids      = ["subnet-xxxxxxxx", "subnet-yyyyyyyy"]
cpu             = 512
memory          = 1024
desired_count   = 2
```

---

## Step 3: Deploy

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Apply changes
terraform apply

# Get the ALB DNS name
terraform output alb_dns_name
```

---

## Step 4: Test the Deployment

```bash
# Health check
curl http://<ALB_DNS_NAME>/health

# Prediction
curl -X POST http://<ALB_DNS_NAME>/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

---

## Auto Scaling

Add to your Terraform configuration:

```hcl
# Auto Scaling Target
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# CPU-based scaling
resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.app_name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70
  }
}
```

---

## Cost Optimization Tips

| Strategy | Savings |
|----------|---------|
| Use Fargate Spot | Up to 70% |
| Right-size tasks | Variable |
| Scale to zero at night | ~33% |
| Use smaller models | Less CPU/memory |

### Fargate Spot Configuration

```hcl
resource "aws_ecs_service" "app" {
  # ... existing config ...

  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 2
  }

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1  # At least 1 on-demand task
  }
}
```

---

## Cleanup

```bash
# Destroy all resources
terraform destroy

# Delete ECR images
aws ecr batch-delete-image \
    --repository-name my-ml-app \
    --image-ids imageTag=latest
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Task keeps restarting | Check CloudWatch logs, increase start period |
| Health check failing | Verify /health endpoint, check security groups |
| OOM errors | Increase memory in task definition |
| Slow startup | Pre-download model in container, use larger instances |
| Connection refused | Check security groups, verify container port |

---

## Next Steps

Continue to Module 05: **Capstone Project**!


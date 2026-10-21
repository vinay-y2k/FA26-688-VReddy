terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1" # Bedrock model access tier zone
}

# ======================================
# Declare variables
# ======================================

variable "db_password" {
  type        = string
  description = "The master password for the Aurora database cluster"
  sensitive   = true # Prevents Terraform from printing the password to your terminal logs
}
# ==========================================
# NETWORK BOUNDARY CONFIGURATION (VPC Subnets)
# ==========================================
resource "aws_vpc" "capstone_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "capstone-hybrid-rag-vpc" }
}

resource "aws_subnet" "subnet_a" {
  vpc_id            = aws_vpc.capstone_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-1a"
}

resource "aws_subnet" "subnet_b" {
  vpc_id            = aws_vpc.capstone_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-east-1b"
}

resource "aws_db_subnet_group" "aurora_subnet_group" {
  name       = "capstone-aurora-subnets"
  subnet_ids = [aws_subnet.subnet_a.id, aws_subnet.subnet_b.id]
}



# =======================================================
# PUBLIC NETWORK GATEWAY INFRASTRUCTURE
# =======================================================
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.capstone_vpc.id
  tags   = { Name = "capstone-internet-gateway" }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.capstone_vpc.id
  route {
    cidr_block = "0.0.0.0/0" # Routes all outbound internet traffic through the gateway
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "capstone-public-route-table" }
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.subnet_a.id
  route_table_id = aws_route_table.public_rt.id
}

resource "aws_route_table_association" "b" {
  subnet_id      = aws_subnet.subnet_b.id
  route_table_id = aws_route_table.public_rt.id
}

# =======================================================
# SECURITY GROUP: Open Port 5432 for Public Testing Loops
# =======================================================
resource "aws_security_group" "db_sg" {
  name   = "capstone-db-security-group"
  vpc_id = aws_vpc.capstone_vpc.id

  # Ingress Rule: Allows your local machine to communicate with the DB engine
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # Allows access from the public internet for rapid testing
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# =======================================================
# UPDATE VECTOR CLUSTER: Apply Parameter Settings
# =======================================================
resource "aws_rds_cluster" "aurora_vector_cluster" {
  cluster_identifier              = "capstone-aurora-serverless-vector-db"
  engine                          = "aurora-postgresql"
  engine_version                  = "16.14"
  database_name                   = "aws_aifootprint"
  master_username                 = "aws_db_admin_root"
  master_password                 = var.db_password
  db_subnet_group_name            = aws_db_subnet_group.aurora_subnet_group.name
  vpc_security_group_ids          = [aws_security_group.db_sg.id]
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.aurora_pg.name
  skip_final_snapshot             = true

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 2.0
  }
}

resource "aws_rds_cluster_instance" "aurora_instances" {
  identifier          = "capstone-vector-instance-1"
  cluster_identifier  = aws_rds_cluster.aurora_vector_cluster.id
  instance_class      = "db.serverless"
  engine              = aws_rds_cluster.aurora_vector_cluster.engine
  engine_version      = aws_rds_cluster.aurora_vector_cluster.engine_version
  
  # 🎯 CRITICAL FIX: Forces AWS to bind a publicly resolvable IP address to the node
  publicly_accessible = true 
}


# Dynamic Scaling Capacity Bounds Configuration (Protects budget limits)
resource "aws_rds_cluster_parameter_group" "aurora_pg" {
  name   = "capstone-aurora-pg"
  family = "aurora-postgresql16"

    # MEMORY CONSTRAINT: Caps worker sort memory to 64MB
  # Forces heavy vector math to process incrementally rather than triggering an ACU scaling surge
  parameter {
    name  = "work_mem"
    value = "65536" 
  }

  # 🛡️ RUNAWAY QUERY TERMINATION CIRCUITS
  # Kills any single vector query or index execution if it locks a thread for more than 15 seconds
  parameter {
    name  = "statement_timeout"
    value = "15000" 
  }


}

# ==========================================
# OUTPUT DEFINITIONS FOR CONSOLE COUPLING
# ==========================================
output "aurora_cluster_endpoint" {
  description = "aws_gateway.py script host parameter"
  value       = aws_rds_cluster.aurora_vector_cluster.endpoint
}

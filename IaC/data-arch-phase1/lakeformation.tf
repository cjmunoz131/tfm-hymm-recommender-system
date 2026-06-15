# ==============================================================================
# LAKE FORMATION PERMISSIONS
# ==============================================================================
# Permisos lógicos a nivel de Lake Formation para que los Crawlers y Glue Jobs
# puedan operar sobre las bases de datos del Catalog sin necesidad de registrar
# físicamente un Data Lake Location.
# ==============================================================================

# ---------------------------------------------------------------
# DATA LAKE ADMINISTRATOR — Registrar al rol de Terraform
# ---------------------------------------------------------------
# PREREQUISITO: Registrar manualmente el rol en Lake Formation Console
# la primera vez (Lake Formation no acepta credenciales temporales).
# Console → Lake Formation → Administrative roles → Add:
#   arn:aws:iam::<ACCOUNT_ID>:role/terraform-role
#
# Una vez registrado, este recurso mantiene el estado en Terraform.
resource "aws_lakeformation_data_lake_settings" "admin" {
  provider = aws.account1

  admins = [
    "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/terraform-role",
    "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-reserved/sso.amazonaws.com/us-east-2/AWSReservedSSO_AdministratorAccess_cabda561aaa68976"
  ]
}

# resource "aws_lakeformation_data_lake_settings" "admin2" {
#   provider = aws.account1

#   admins = [
#     "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-reserved/sso.amazonaws.com/us-east-2/AWSReservedSSO_AdministratorAccess_cabda561aaa68976"
#   ]
# }

# ---------------------------------------------------------------
# CRAWLER TMDB (Bronze) — Permisos sobre DB + todas las tablas
# ---------------------------------------------------------------
resource "aws_lakeformation_permissions" "crawler_tmdb_database" {
  provider   = aws.account1
  principal  = module.aws-data-governance-metadata-tmdb-crawler-glue-layer-module.role_default_arn
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["CREATE_TABLE", "DESCRIBE", "ALTER"]

  database {
    name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "crawler_tmdb_all_tables" {
  provider   = aws.account1
  principal  = module.aws-data-governance-metadata-tmdb-crawler-glue-layer-module.role_default_arn
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["ALL"]

  table {
    database_name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
    wildcard      = true
  }
}

# ---------------------------------------------------------------
# CRAWLER ML (Bronze) — Permisos sobre DB + todas las tablas
# ---------------------------------------------------------------
resource "aws_lakeformation_permissions" "crawler_ml_database" {
  provider   = aws.account1
  principal  = module.aws-data-governance-metadata-ml-crawler-glue-layer-module.role_default_arn
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["CREATE_TABLE", "DESCRIBE", "ALTER"]

  database {
    name = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "crawler_ml_all_tables" {
  provider   = aws.account1
  principal  = module.aws-data-governance-metadata-ml-crawler-glue-layer-module.role_default_arn
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["ALL"]

  table {
    database_name = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
    wildcard      = true
  }
}

# ---------------------------------------------------------------
# GLUE JOB: TMDB Baseline Bronze
# Acceso: lee de ml_bronze (links), escribe en tmdb_bronze (movies)
# ---------------------------------------------------------------
resource "aws_lakeformation_permissions" "glue_tmdb_baseline_ml_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_tmdb_baseline_bronze_table_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["DESCRIBE"]

  database {
    name = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "glue_tmdb_baseline_ml_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_tmdb_baseline_bronze_table_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["SELECT", "DESCRIBE"]

  table {
    database_name = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
    wildcard      = true
  }
}

resource "aws_lakeformation_permissions" "glue_tmdb_baseline_tmdb_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_tmdb_baseline_bronze_table_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["DESCRIBE", "CREATE_TABLE", "ALTER"]

  database {
    name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "glue_tmdb_baseline_tmdb_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_tmdb_baseline_bronze_table_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["ALL"]

  table {
    database_name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
    wildcard      = true
  }
}

# ---------------------------------------------------------------
# GLUE JOB: TMDB Posters Silver
# Acceso: lee de tmdb_bronze (movies)
# ---------------------------------------------------------------
resource "aws_lakeformation_permissions" "glue_posters_tmdb_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_tmdb_posters_silver_image_modal_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["DESCRIBE"]

  database {
    name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "glue_posters_tmdb_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_tmdb_posters_silver_image_modal_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["SELECT", "DESCRIBE"]

  table {
    database_name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
    wildcard      = true
  }
}

# ---------------------------------------------------------------
# GLUE JOB: Create OBT Silver Tables
# Acceso: lee de ml_bronze (ratings, links) + tmdb_bronze (movies)
#          escribe en silver database (obt_movie_affinity)
# ---------------------------------------------------------------
resource "aws_lakeformation_permissions" "glue_obt_ml_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["DESCRIBE"]

  database {
    name = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "glue_obt_ml_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["SELECT", "DESCRIBE"]

  table {
    database_name = module.aws_data_governance_catalog_bronze_ml_database_glue_layer_module.name
    wildcard      = true
  }
}

resource "aws_lakeformation_permissions" "glue_obt_tmdb_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["DESCRIBE"]

  database {
    name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "glue_obt_tmdb_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["SELECT", "DESCRIBE"]

  table {
    database_name = module.aws_data_governance_catalog_bronze_tmdb_database_glue_layer_module.name
    wildcard      = true
  }
}

resource "aws_lakeformation_permissions" "glue_obt_silver_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["DESCRIBE", "CREATE_TABLE", "ALTER"]

  database {
    name = module.aws_data_governance_catalog_silver_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "glue_obt_silver_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_create_obt_tables_silver_layer_module.iam_roles
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["ALL"]

  table {
    database_name = module.aws_data_governance_catalog_silver_database_glue_layer_module.name
    wildcard      = true
  }
}

# ---------------------------------------------------------------
# ACCESO SSO ADMIN — SELECT sobre tablas Silver para Athena
# ---------------------------------------------------------------
resource "aws_lakeformation_permissions" "sso_admin_silver_database" {
  provider   = aws.account1
  principal  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-reserved/sso.amazonaws.com/us-east-2/AWSReservedSSO_AdministratorAccess_cabda561aaa68976"
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["DESCRIBE", "ALTER", "CREATE_TABLE"]

  database {
    name = module.aws_data_governance_catalog_silver_database_glue_layer_module.name
  }
}

resource "aws_lakeformation_permissions" "sso_admin_silver_all_tables" {
  provider   = aws.account1
  principal  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/aws-reserved/sso.amazonaws.com/us-east-2/AWSReservedSSO_AdministratorAccess_cabda561aaa68976"
  depends_on = [aws_lakeformation_data_lake_settings.admin]

  permissions = ["ALL"]

  table {
    database_name = module.aws_data_governance_catalog_silver_database_glue_layer_module.name
    wildcard      = true
  }
}

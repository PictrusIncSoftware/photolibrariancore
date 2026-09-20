        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            file_size INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_extension TEXT,
            created_timestamp INTEGER NOT NULL,
            modified_timestamp INTEGER NOT NULL,

            -- Camera/capture metadata
            camera_make TEXT,
            camera_model TEXT,
            lens_model TEXT,
            focal_length REAL,
            aperture REAL,
            shutter_speed REAL,
            iso INTEGER,
            capture_datetime TEXT,

            -- Image properties
            pixel_width INTEGER,
            pixel_height INTEGER,
            color_space TEXT,
            bit_depth INTEGER,

            -- GPS coordinates
            gps_latitude REAL,
            gps_longitude REAL,
            gps_altitude REAL,

            -- IPTC/copyright
            copyright TEXT,
            creator TEXT,
            description TEXT,

            -- Organization metadata
            rating INTEGER,
            flag TEXT,
            color_label TEXT,

            -- Indices for common queries
            indexed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for efficient filtering
        CREATE INDEX IF NOT EXISTS idx_camera_model ON images(camera_model);
        CREATE INDEX IF NOT EXISTS idx_capture_datetime ON images(capture_datetime);
        CREATE INDEX IF NOT EXISTS idx_created_timestamp ON images(created_timestamp);
        CREATE INDEX IF NOT EXISTS idx_file_extension ON images(file_extension);
        CREATE INDEX IF NOT EXISTS idx_rating ON images(rating);
        CREATE INDEX IF NOT EXISTS idx_flag ON images(flag);
        CREATE INDEX IF NOT EXISTS idx_color_label ON images(color_label);

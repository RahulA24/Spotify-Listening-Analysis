import pandas as pd
import re

class SpotifyAgent:
    def __init__(self, df):
        self.df = df

        # Ensure timestamp column is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df["ts"]):
            self.df["ts"] = pd.to_datetime(self.df["ts"])

        min_date = self.df["ts"].min().strftime("%B %Y")
        max_date = self.df["ts"].max().strftime("%B %Y")

        print("Spotify Data Agent Online.")
        print(f"Data Coverage: **{min_date}** to **{max_date}**")
        print("Ready for specific dates (for example: 'Nov 24 2023' or 'Saturdays in 2024').")

        # Month name to number map
        self.months = {
            "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
            "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
            "nov": 11, "november": 11, "dec": 12, "december": 12
        }

        # Day name to weekday number (0 = Monday, 6 = Sunday)
        self.days_map = {
            "mon": 0, "monday": 0, "tue": 1, "tuesday": 1, "wed": 2, "wednesday": 2,
            "thu": 3, "thursday": 3, "fri": 4, "friday": 4, "sat": 5, "saturday": 5,
            "sun": 6, "sunday": 6
        }

    def _extract_date(self, query):
        """Find numeric day, month, year, and weekday name from free text."""
        query = query.lower()

        # Year like 2019, 2020
        year_match = re.search(r"\b(20[1-2][0-9])\b", query)
        year = int(year_match.group(1)) if year_match else None

        # Month by name
        month = None
        for name, num in self.months.items():
            if name in query:
                month = num
                break

        # Numeric day, avoid false positives like "Top 5"
        day = None
        day_match = re.search(r"\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\b", query)
        if day_match:
            potential_day = int(day_match.group(1))
            if "top" in query and str(potential_day) in query and not month:
                day = None
            else:
                day = potential_day

        # Weekday name detection (match whole words only)
        weekday = None
        weekday_name = None
        for name, num in self.days_map.items():
            if re.search(r"\b" + name + r"\b", query):
                weekday = num
                weekday_name = name.capitalize()
                break

        return day, month, year, weekday, weekday_name

    def chat(self, user_query):
        """Parse a natural language question and return a short insight."""
        user_query = user_query.lower()

        # Extract date filters and make a working copy of the data
        day, month, year, weekday, weekday_name = self._extract_date(user_query)
        filtered_df = self.df.copy()

        # Build a human label for the time window
        parts = []
        if weekday_name:
            parts.append(f"{weekday_name}s")
        if day:
            parts.append(str(day))
        if month:
            parts.append(pd.to_datetime(f"2022-{month}-1").strftime("%B"))
        if year:
            parts.append(str(year))
        time_label = " ".join(parts) if parts else "All Time"

        # Apply date filters one by one
        if year:
            filtered_df = filtered_df[filtered_df["ts"].dt.year == year]
        if month:
            filtered_df = filtered_df[filtered_df["ts"].dt.month == month]
        if day:
            filtered_df = filtered_df[filtered_df["ts"].dt.day == day]
        if weekday is not None:
            filtered_df = filtered_df[filtered_df["ts"].dt.dayofweek == weekday]

        if filtered_df.empty:
            return f"No data found for {time_label}."

        # Specific question handlers

        # Total listening time in minutes
        if "time" in user_query and ("listen" in user_query or "mins" in user_query):
            total_ms = filtered_df["ms_played"].sum()
            minutes = int(total_ms / 60000)
            return f"Time Listened ({time_label}): **{minutes:,} minutes**."

        # Top artist by play count
        elif "top artist" in user_query:
            if "master_metadata_album_artist_name" not in filtered_df.columns:
                return "Error: Data missing."
            top = filtered_df["master_metadata_album_artist_name"].mode()[0]
            count = filtered_df["master_metadata_album_artist_name"].value_counts().iloc[0]
            return f"Top Artist ({time_label}): **{top}** ({count} plays)."

        # Top song and its artist
        elif "top song" in user_query:
            if "master_metadata_track_name" not in filtered_df.columns:
                return "Error: Data missing."
            song = filtered_df["master_metadata_track_name"].mode()[0]
            count = filtered_df["master_metadata_track_name"].value_counts().iloc[0]
            artist = filtered_df[filtered_df["master_metadata_track_name"] == song][
                "master_metadata_album_artist_name"
            ].iloc[0]
            return f"Top Song ({time_label}): **{song}** by {artist} ({count} plays)."

        # Top 5 songs list
        elif "top 5" in user_query:
            top_5 = filtered_df["master_metadata_track_name"].value_counts().head(5)
            response = f"**Top 5 Songs ({time_label}):**\n"
            for song, count in top_5.items():
                response += f"- {song}: {count}\n"
            return response

        # When do I listen? Peak hour, optionally best day/month
        elif "when" in user_query:
            if "hour" not in filtered_df.columns:
                return "Error: Time data missing."

            peak_hour = int(filtered_df["hour"].mode()[0])
            period = "AM" if peak_hour < 12 else "PM"
            friendly_hour = peak_hour if peak_hour <= 12 else peak_hour - 12
            if friendly_hour == 0:
                friendly_hour = 12

            response = f"**Peak Listening Time ({time_label}):** Around **{friendly_hour} {period}**."

            # Only show best day if the query is not already constrained to a day or weekday
            if not day and weekday is None:
                peak_day = filtered_df["ts"].dt.day_name().mode()[0]
                response += f"\n- **Best Day:** {peak_day}"

            if not month and not day:
                peak_month = filtered_df["ts"].dt.month_name().mode()[0]
                response += f"\n- **Best Month:** {peak_month}"

            return response

        # Weekend-specific insight
        elif "weekend" in user_query:
            weekend_df = filtered_df[filtered_df["is_weekend"] == 1]
            if weekend_df.empty:
                return "No weekend data."
            top = weekend_df["master_metadata_album_artist_name"].mode()[0]
            return f"Weekend Vibe ({time_label}): **{top}**."

        # Skip rate insights
        elif "skip" in user_query:
            if "is_skipped" not in filtered_df.columns:
                return "No skip data."
            stats = filtered_df.groupby("master_metadata_album_artist_name")["is_skipped"].mean()
            counts = filtered_df["master_metadata_album_artist_name"].value_counts()
            valid = stats[counts > 5]
            if valid.empty:
                return "Not enough data."
            worst = valid.idxmax()
            return f"⏭Most Skipped ({time_label}): **{worst}** ({valid[worst]:.1%} skip rate)."

        # Fallback when the intent is unclear
        else:
            return f"I can filter by {time_label}, but I didn't catch the question."

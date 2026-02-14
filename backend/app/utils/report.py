"""Utility module for visualization and additional analysis functions."""

import folium
from folium.plugins import HeatMap, MarkerCluster


def create_safety_map(lat, lon, nearby_crimes, analysis_result):
    """Create an interactive safety map with crime data."""
    # Create base map centered on location
    safety_map = folium.Map(location=[lat, lon], zoom_start=14)

    # Add marker for specified location
    folium.Marker(
        [lat, lon],
        popup=f"Safety Score: {analysis_result['safety_score']}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(safety_map)

    # Add crime markers
    if not nearby_crimes.empty:
        # Create marker cluster for crimes
        marker_cluster = MarkerCluster().add_to(safety_map)

        # Add individual crime markers
        for _, crime in nearby_crimes.iterrows():
            folium.Marker(
                [crime["latitude"], crime["longitude"]],
                popup=(
                    f"Type: {crime['crime_type']}<br>"
                    f"Distance: {crime['distance']:.2f} km"
                ),
                icon=folium.Icon(color="red", icon="warning-sign", prefix="fa"),
            ).add_to(marker_cluster)

        # Add heatmap layer
        heat_data = [
            [row["latitude"], row["longitude"]] for _, row in nearby_crimes.iterrows()
        ]
        HeatMap(heat_data, radius=15).add_to(safety_map)

    # Create a circle showing the safety score
    folium.Circle(
        location=[lat, lon],
        radius=500,  # meters
        popup=f"Safety Score: {analysis_result['safety_score']}",
        color=(
            "green"
            if analysis_result["safety_score"] > 70
            else "orange" if analysis_result["safety_score"] > 40 else "red"
        ),
        fill=True,
        fill_opacity=0.2,
    ).add_to(safety_map)

    return safety_map


def get_demographic_context(amenity_type):
    """Determine the demographic group and specific safety concerns based on amenity type."""
    amenity_type_lower = amenity_type.lower()

    if any(word in amenity_type_lower for word in ["school", "daycare", "kindergarten", "playground"]):
        return {
            "group": "children and families",
            "concerns": [
                "child safety",
                "predatory crimes",
                "traffic safety",
                "drug activity",
            ],
            "focus": "protecting minors",
        }
    if any(word in amenity_type_lower for word in ["senior", "nursing", "retirement", "elderly"]):
        return {
            "group": "elderly residents",
            "concerns": ["assault", "robbery", "scams", "accessibility"],
            "focus": "vulnerable adult protection",
        }
    if any(word in amenity_type_lower for word in ["hospital", "clinic", "medical"]):
        return {
            "group": "patients and healthcare workers",
            "concerns": [
                "theft",
                "assault",
                "parking safety",
                "emergency access",
            ],
            "focus": "healthcare facility security",
        }
    if any(word in amenity_type_lower for word in ["bar", "nightclub", "pub"]):
        return {
            "group": "nightlife patrons",
            "concerns": [
                "assault",
                "DWI",
                "harassment",
                "late-night crimes",
            ],
            "focus": "evening and night safety",
        }
    if any(word in amenity_type_lower for word in ["restaurant", "cafe", "food"]):
        return {
            "group": "diners and visitors",
            "concerns": ["theft", "harassment", "property crime"],
            "focus": "general public safety",
        }
    if any(word in amenity_type_lower for word in ["park", "recreation"]):
        return {
            "group": "families and outdoor enthusiasts",
            "concerns": ["assault", "theft", "vandalism", "drug activity"],
            "focus": "public space safety",
        }
    return {
        "group": "general public",
        "concerns": ["crime", "safety"],
        "focus": "overall area security",
    }


def generate_safety_report(analysis_result, address_str=None, amenity_type=None):
    """Generate a detailed safety report in text format with demographic-specific analysis."""
    safety_score = analysis_result["safety_score"]

    # Get demographic context if amenity type provided
    demo_context = get_demographic_context(amenity_type) if amenity_type else None
    # Determine safety level
    if safety_score >= 80:
        safety_level = "Very Safe"
    elif safety_score >= 60:
        safety_level = "Safe"
    elif safety_score >= 40:
        safety_level = "Moderate"
    elif safety_score >= 20:
        safety_level = "Concerning"
    else:
        safety_level = "High Risk"

    # Build report
    report = []
    report.append("📍 SAFETY ANALYSIS")
    if address_str:
        report.append(f"Location: {address_str}")

    report.append(f"\n🛡️ SAFETY SCORE: {safety_score}/100 ({safety_level})")

    # Only show top 3 crimes to reduce clutter
    report.append("\n⚠️ TOP CRIME CONCERNS:")
    for i, (crime, count) in enumerate(
        list(analysis_result["common_crimes"].items())[:3], start=1
    ):
        report.append(f"  {i}. {crime}: {count} incidents")
    
    report.append("\nNEARBY ACTIVITY")
    report.append(f"Total Nearby Crimes: {analysis_result['nearby_crime_count']}")

    if "time_analysis" in analysis_result:
        time_data = analysis_result["time_analysis"]["time_of_day"]
        total = sum(time_data.values())

        if total > 0:
            # Only show highest risk time
            high_risk = max(time_data.items(), key=lambda x: x[1])[0]
            high_risk_count = time_data[high_risk]
            percentage = (high_risk_count / total) * 100
            report.append(
                f"\n⏰ HIGHEST RISK TIME: {high_risk.capitalize()} ({percentage:.0f}% of crimes)"
            )

    if "amenities" in analysis_result:
        report.append("\nNEARBY AMENITIES")
        report.append(
            f"Total amenities within 1 km: {analysis_result['amenities']['nearby_count']}"
        )

        if analysis_result["amenities"]["type_counts"]:
            report.append("Amenities by type:")
            for amenity_type, count in analysis_result["amenities"][
                "type_counts"
            ].items():
                report.append(f"  - {amenity_type.capitalize()}: {count}")

    # Add demographic-specific analysis
    if demo_context:
        report.append(f"\n👥 FOR {demo_context['group'].upper()}")

        # Check for relevant crimes - only show if found
        crime_types = analysis_result.get("crime_types", {})
        relevant_crimes = []
        for concern in demo_context["concerns"]:
            for crime, count in crime_types.items():
                if concern.lower() in crime.lower() or crime.lower() in concern.lower():
                    relevant_crimes.append((crime, count))

        if relevant_crimes:
            report.append("⚠️ Specific Concerns:")
            for crime, count in relevant_crimes[:3]:  # Top 3 only
                report.append(f"  • {crime}: {count} incidents")

    report.append("\n💡 SAFETY RECOMMENDATIONS")

    # Demographic-specific recommendations
    if demo_context:
        if "children" in demo_context["group"]:
            if safety_score < 60:
                report.append("⚠️ CAUTION FOR CHILDREN:")
                report.append("• Consider alternative locations")
                report.append("• Ensure constant adult supervision")
                report.append("• Avoid isolated areas")
            else:
                report.append("✓ SAFE FOR CHILDREN with precautions:")
                report.append("• Maintain supervision during activities")
                report.append("• Teach basic safety awareness")

        elif "elderly" in demo_context["group"]:
            if safety_score < 60:
                report.append("⚠️ CAUTION FOR ELDERLY:")
                report.append("• Use buddy system")
                report.append("• Consider transportation services")
            else:
                report.append("✓ SUITABLE FOR ELDERLY with precautions:")
                report.append("• Ensure good lighting")
                report.append("• Keep emergency contacts ready")

        elif "nightlife" in demo_context["group"]:
            if safety_score < 60:
                report.append("⚠️ NIGHTLIFE CAUTION:")
                report.append("• Travel in groups only")
                report.append("• Use rideshare services")
            else:
                report.append("✓ SAFE FOR NIGHTLIFE with precautions:")
                report.append("• Exercise caution during late hours")
                report.append("• Share location with friends")

    # General recommendations - condensed
    if safety_score < 40:
        report.append("\n🚨 HIGH RISK AREA:")
        report.append("• Exercise extreme caution")
        report.append("• Avoid walking alone")
    elif safety_score < 70:
        report.append("\n⚠️ MODERATE RISK:")
        report.append("• Stay aware of surroundings")
        report.append("• Take normal precautions")
    else:
        report.append("\n✓ RELATIVELY SAFE:")
        report.append("• Standard precautions recommended")

    return "\n".join(report)

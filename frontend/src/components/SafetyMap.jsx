import { MapContainer, TileLayer, CircleMarker, LayersControl, LayerGroup, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import { useEffect } from 'react'

// Fix default marker icon issue with Leaflet in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

	const SafetyMap = ({ clusters, center }) => {
	  const clusterColors = {
	    dbscan: '#ec4899',
	    demographic: '#db2777',
	    density: '#be185d'
	  }

	  useEffect(() => {
	    // Force map resize on mount
	    setTimeout(() => {
	      window.dispatchEvent(new Event('resize'));
	    }, 100);
	  }, []);

	  if (!center || !clusters) {
	    return (
	      <div className="map-container" style={{ height: '100%', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
	        <p>Loading map...</p>
	      </div>
	    );
	  }

	  return (
	    <div className="map-container" style={{ height: '100%', width: '100%' }}>
	      <MapContainer 
	        center={center} 
	        zoom={13} 
	        scrollWheelZoom={true}
	        style={{ height: '100%', width: '100%' }}
	      >
	        <TileLayer
	          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
	          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
	        />
	        
	        {/* Main location marker */}
	        <Marker position={center}>
	          <Popup>
	            <strong>Your Queried Location</strong><br />
	            Lat: {center[0].toFixed(4)}, Lon: {center[1].toFixed(4)}
	          </Popup>
	        </Marker>
	        
	        <LayersControl position="topright">
	          {clusters.dbscan && clusters.dbscan.length > 0 && (
	            <LayersControl.Overlay name="DBSCAN Clusters" checked>
	              <LayerGroup>
	                {clusters.dbscan.map((cluster, idx) => (
	                  <CircleMarker
	                    key={`dbscan-${idx}`}
	                    center={[cluster.center_lat, cluster.center_lon]}
	                    radius={8}
	                    pathOptions={{
	                      color: clusterColors.dbscan,
	                      fillColor: clusterColors.dbscan,
	                      fillOpacity: 0.6,
	                      weight: 2
	                    }}
	                  />
	                ))}
	              </LayerGroup>
	            </LayersControl.Overlay>
	          )}
	
	          {clusters.demographic && clusters.demographic.length > 0 && (
	            <LayersControl.Overlay name="Demographic Zones">
	              <LayerGroup>
	                {clusters.demographic.map((zone, idx) => (
	                  <CircleMarker
	                    key={`demo-${idx}`}
	                    center={[zone.center_lat, zone.center_lon]}
	                    radius={8}
	                    pathOptions={{
	                      color: clusterColors.demographic,
	                      fillColor: clusterColors.demographic,
	                      fillOpacity: 0.6,
	                      weight: 2
	                    }}
	                  />
	                ))}
	              </LayerGroup>
	            </LayersControl.Overlay>
	          )}
	
	          {clusters.density && clusters.density.length > 0 && (
	            <LayersControl.Overlay name="Density Zones">
	              <LayerGroup>
	                {clusters.density.map((density, idx) => (
	                  <CircleMarker
	                    key={`density-${idx}`}
	                    center={[density.latitude, density.longitude]}
	                    radius={8}
	                    pathOptions={{
	                      color: clusterColors.density,
	                      fillColor: clusterColors.density,
	                      fillOpacity: 0.4,
	                      weight: 2
	                    }}
	                  />
	                ))}
	              </LayerGroup>
	            </LayersControl.Overlay>
	          )}
	        </LayersControl>
      </MapContainer>
    </div>
  )
}

export default SafetyMap